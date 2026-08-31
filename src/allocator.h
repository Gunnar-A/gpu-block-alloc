#pragma once
#include "intr/mod.h"
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>
#else
    #define __host__
    #define __device__
#endif

typedef unsigned long long int defaultAllocMaskElem;
typedef unsigned long long int defaultSlabAddr;

const size_t DEFAULT_SLAB_SIZE = 1 << 16;       // 64KB

template <size_t n>
struct Size {
    static const size_t VALUE = n;
};

template <typename type>
struct AdrInfo {
    __host__ __device__
    static type null(){
        return static_cast<type>(-1);
    }
};


// basic slab structure
template <size_t SLAB_SIZE = DEFAULT_SLAB_SIZE, typename ELEM_TYPE = defaultAllocMaskElem>
struct Slab {
    typedef ELEM_TYPE allocMaskElem;
    static const size_t ELEM_SIZE = sizeof(ELEM_TYPE);
    static const size_t SIZE = SLAB_SIZE;
    static const size_t ELEM_COUNT = (SLAB_SIZE + ELEM_SIZE - 1) / ELEM_SIZE;

    ELEM_TYPE data[ELEM_COUNT];
}; 

// slab proxy  
template <typename SLAB_TYPE>
struct defaultSlabProxy {
    typedef SLAB_TYPE slabType;
    typedef typename SLAB_TYPE::allocMaskElem allocMaskElem;
    typedef unsigned long long int AllocState;

    static const AllocState OFFSET = sizeof(AllocState) * 8 / 2;
    static const AllocState COUNT_MASK = (((AllocState)1) << OFFSET) - (AllocState)1;
    static const AllocState SIZE_MASK = ~COUNT_MASK;
    
    static const size_t SLAB_SIZE = SLAB_TYPE::SIZE;
    static const size_t SLAB_ELEM_COUNT = SLAB_TYPE::ELEM_COUNT;
    static const size_t SLAB_ELEM_BIT_SIZE = sizeof(allocMaskElem) * 8;         // check sizeof

    enum SlabState : uint32_t   {
        FREE = 0,
        PARTIAL = 1,
        FULL = 2
    };

    allocMaskElem allocMask;
    AllocState allocState;
    uint32_t reservationState;
    unsigned long long int next;


    __host__ __device__
    size_t getSize()
    { return (intr::atomic::load_acquire(&allocState) & SIZE_MASK) >> OFFSET;}

    __host__ __device__
    size_t getCount()
    { return intr::atomic::load_acquire(&allocState) & COUNT_MASK;}

    __host__ __device__
    size_t isEmpty()
    { return getCount() == 0; }

    __host__ __device__
    bool isFull(){
        AllocState currentState = intr::atomic::load_acquire(&allocState);
        size_t objSize = (currentState & SIZE_MASK) >> OFFSET;
        if(objSize == 0)
            return false;

        size_t currentCount = currentState & COUNT_MASK;
        size_t maxCount = slabObjCount(objSize);
                
        return (currentCount >= maxCount);
    } // end of isFull
    
    __host__ __device__
    bool isAvailable(){
        uint32_t state = intr::atomic::load_acquire(&reservationState);
        return (state == PARTIAL && !isFull());
    }

    // atomics..
    __host__ __device__
    bool tryClaim(SLAB_TYPE* slab, size_t objectSz){
        AllocState expected = 0;
        AllocState newState = ((AllocState)objectSz) << OFFSET;

        AllocState old = intr::atomic::CAS_acq_rel(&allocState, expected, newState);
        if(old == expected){
            size_t maxObjCount = slabObjCount(objectSz);
            if (maxObjCount <= SLAB_ELEM_BIT_SIZE)
                intr::atomic::store_relaxed(&allocMask, static_cast<allocMaskElem>(0));
            else {
                size_t maskElemCount = (maxObjCount + SLAB_ELEM_BIT_SIZE - 1) / SLAB_ELEM_BIT_SIZE;
                for (int i = 0; i < maskElemCount; i++)
                    intr::atomic::store_relaxed(&slab->data[i], static_cast<allocMaskElem>(0));
            }
            intr::atomic::store_release(&reservationState, static_cast<uint32_t>(PARTIAL));
            printf("Successfully claimed\n");
            return true;
        }

        size_t currentSz = (old & SIZE_MASK) >> OFFSET;
        uint32_t state = intr::atomic::load_acquire(&reservationState);

        return (currentSz == objectSz) && (state == PARTIAL);
    } // end of tryClaim

    __host__ __device__
    bool clearAllocState(){
        AllocState current = intr::atomic::load_acquire(&allocState);

        while(true){
            size_t count = current & COUNT_MASK;
            if(count != 0)
                return false;
            size_t size = (current & SIZE_MASK) >> OFFSET;
            if(size == 0)
                return false;

            AllocState old = intr::atomic::CAS_acq_rel(&allocState, current, 0ull);
            if(old == current){
                intr::atomic::store_release(&reservationState, static_cast<uint32_t>(FREE));
                return true;
            }
            current = old;
        }// end while
    } // end 

    __host__ __device__
    size_t slabObjCount(size_t objectSize){
        if(objectSize <= 0)
            return 0;           // lets say yes...

        size_t maxObjects = SLAB_SIZE / objectSize; // 4096/32 -> 128
        if(maxObjects <= SLAB_ELEM_BIT_SIZE)
            return maxObjects;
        
        size_t maskElem = (maxObjects + SLAB_ELEM_BIT_SIZE - 1) / SLAB_ELEM_BIT_SIZE; // -> 2
        size_t maskOverhead = (maskElem > 1) ? maskElem * sizeof(allocMaskElem) : 0; // 2 * 8 -> 16

        if(maskOverhead >= SLAB_SIZE)
            return 0;

        size_t openBytes = SLAB_SIZE - maskOverhead; // 4096 - 16 -> 4080
        size_t realObjs = openBytes / objectSize; // 127

        return realObjs;
    } // end of slabObjCount

    
    __host__ __device__
    bool tryAllocateBit(allocMaskElem& elem, size_t& bitIndex){
        for(size_t attempts = 0; attempts < 64; attempts++){
            // fresh loak each iter
            allocMaskElem currentMask = intr::atomic::load_acquire(&elem);

            allocMaskElem freeMask = ~currentMask;
            if(freeMask == 0)
                return false;

            #if defined(__CUDA_ARCH__)
                int ff = __ffsll(static_cast<unsigned long long>(freeMask));
                if(ff == 0)
                    return false;
                size_t target = static_cast<size_t>(ff - 1);
            #else
                size_t target = static_cast<size_t>(__builtin_ctzll(static_cast<unsigned long long>(freeMask)));
            #endif

            if(target >= SLAB_ELEM_BIT_SIZE)
                return false;

            allocMaskElem targetBit = ((allocMaskElem)1) << target;
            allocMaskElem newMask = currentMask | targetBit;

            allocMaskElem old = intr::atomic::CAS_acq_rel(&elem, currentMask, newMask);
            if(old == currentMask){
                bitIndex = target;
                return true;
            }
        }
        return false;
    } // end of tryAllocBit


    __host__ __device__
    void* indexToPtr(SLAB_TYPE* slab, size_t objectSize, size_t maskElemCount, size_t ind){
        char* bytePtr = static_cast<char*>(static_cast<void*>(slab));
        if (maskElemCount > 1)
            bytePtr += sizeof(allocMaskElem) * maskElemCount;
        bytePtr += objectSize * ind;

        return static_cast<void*>(bytePtr);
    } // end

    __host__ __device__
    void* alloc(SLAB_TYPE* slab, bool& slabFilled){
        // just in case...
        uint32_t currentReservState = intr::atomic::load_acquire(&reservationState);
        if(currentReservState == FULL)
            return nullptr;
        
        AllocState currentState = intr::atomic::load_acquire(&allocState);
        size_t objectSz = (currentState & SIZE_MASK) >> OFFSET;
        size_t currentCount = currentState & COUNT_MASK;
        size_t maxObjCount = slabObjCount(objectSz);

        if(maxObjCount == 0)
            return nullptr;

        // check capacity BEFORE tryAlloc
        if(currentCount >= maxObjCount){
            intr::atomic::store_release(&reservationState, static_cast<uint32_t>(FULL));
            return nullptr;
        }

        // resrver count slot..
        AllocState newState = intr::atomic::add_acq_rel(&allocState, (AllocState)1);
        size_t newCount = (newState & COUNT_MASK)+1;

        // check!
        if(newCount > maxObjCount){
            // rollback
            intr::atomic::add_acq_rel(&allocState, ((AllocState)0) - ((AllocState)1));
            intr::atomic::store_release(&reservationState, static_cast<uint32_t>(FULL));
            return nullptr;
        }

        size_t maskElemCount = (maxObjCount + SLAB_ELEM_BIT_SIZE - 1) / SLAB_ELEM_BIT_SIZE;

        // try to allocate a bit..
        size_t bitIndex;
        bool bitAllocated = false;
        size_t globalIndex = 0;

        bool from_slab_mask = false;
        if(maxObjCount <= SLAB_ELEM_BIT_SIZE){
            // single mask elem
            if(tryAllocateBit(allocMask, bitIndex)){
                bitAllocated = true;
                globalIndex = bitIndex;
            }
            from_slab_mask = true;
        } else {
            // multi mask elements
            for(size_t i = 0; i < maskElemCount; i++){
                if(tryAllocateBit(slab->data[i], bitIndex)){
                    bitAllocated = true;
                    globalIndex = SLAB_ELEM_BIT_SIZE * i + bitIndex;
                    break;
                }
            }
        }
        
        if(!bitAllocated) {
            // rollback count
            intr::atomic::add_acq_rel(&allocState, ((AllocState)0) - ((AllocState)1));
            //printf(":( - No bits available - maxObj:%zu maskElems:%zu (alloc)\n", maxObjCount, maskElemCount);
            return nullptr;
        }

        // update slab state
        slabFilled = (newCount == maxObjCount);
        if(slabFilled)
            intr::atomic::store_release(&reservationState, static_cast<uint32_t>(FULL));


        size_t indMax = (sizeof(SLAB_TYPE)-sizeof(allocMaskElem)*maskElemCount) / objectSz;
        if ( (globalIndex >= indMax) && (!from_slab_mask) ) {
            printf("\n\n(BAD : %llu >= %llu (%d))\n\n",globalIndex,indMax,(int)from_slab_mask);
        }

        void* result = indexToPtr(slab, objectSz, maskElemCount, globalIndex);

        return result;
    } // end of alloc

    __host__ __device__
    bool free(SLAB_TYPE* slab, void* objPtr, bool& slabEmptied, bool& slabWasFull){
        if(!objPtr)
            return false;
        
        char* objBytePtr = static_cast<char*>(objPtr);
        char* slabBytePtr = static_cast<char*>(static_cast<void*>(slab));
        size_t byteOffset = objBytePtr - slabBytePtr;

        AllocState currentState = intr::atomic::load_acquire(&allocState);
        size_t objectSz = (currentState & SIZE_MASK) >> OFFSET;
        size_t currentCount = currentState & COUNT_MASK;
        size_t maxObjCount = slabObjCount(objectSz);

        // nothing allocated
        if(currentCount == 0)
            return false;

        // calc object index
        size_t maskCount = (maxObjCount + SLAB_ELEM_BIT_SIZE - 1) / SLAB_ELEM_BIT_SIZE;
        size_t maskSize = maskCount * sizeof(allocMaskElem);
        size_t firstObjOffset = (maskCount > 1) ? maskSize : 0;

        // invalid ptr
        if(byteOffset < firstObjOffset)
            return false;

        // misaligned
        size_t objOffset = byteOffset - firstObjOffset;
        if(objOffset % objectSz != 0)
            return false;

        // out of range
        size_t objIndex = objOffset / objectSz;
        if(objIndex >= maxObjCount)
            return false;

        // clear the bit first
        size_t maskIndex = objIndex / SLAB_ELEM_BIT_SIZE;
        size_t targetBit = objIndex % SLAB_ELEM_BIT_SIZE;
        allocMaskElem targetMask = ((allocMaskElem)1) << targetBit;
        allocMaskElem prevMask;

        if(maskCount <= 1) {
            prevMask = intr::atomic::and_acq_rel(&allocMask, ~targetMask);
        } else {
            prevMask = intr::atomic::and_acq_rel(&(slab->data[maskIndex]), ~targetMask);
        }

        // check if bit was actually set
        bool wasAllocated = ((prevMask & targetMask) != 0);
        if(!wasAllocated) 
            return false; 

        // decrement count after clearing bit
        AllocState prev = intr::atomic::add_acq_rel(&allocState, ((AllocState)0) - ((AllocState)1));
        AllocState prevCount = (prev & COUNT_MASK);
        AllocState newCount = prevCount - 1;
        
        slabEmptied = (newCount == 0);
        slabWasFull = (prevCount == maxObjCount);
        if(!slabEmptied) {
            uint32_t state = intr::atomic::load_acquire(&reservationState);
            if(state == FULL) 
                intr::atomic::store_release(&reservationState, static_cast<uint32_t>(PARTIAL));
        }
        
        return true;
    } // end of free

}; // end of slabProxy



template <typename IndexType>
struct List {
    public:
    IndexType head;
};

template <typename IndexType>
class Link {
    public:
    IndexType data;

    static const int OFFSET = sizeof(IndexType) * 8 - 1;
    static const IndexType INDEX_MASK = (static_cast<IndexType>(1) << OFFSET) - 1;

    __host__ __device__
    Link(IndexType i, bool d)
        : data((static_cast<IndexType>(d) << OFFSET) | (i & INDEX_MASK)) {}

    __host__ __device__
    static IndexType makeMarkedRaw(const IndexType raw) {
        return raw | (static_cast<IndexType>(1) << OFFSET);
    }

    __host__ __device__
    bool operator==(const Link<IndexType> &other) const { return (data == other.data); }

    __host__ __device__
    bool operator!=(const Link<IndexType> &other) const { return (data != other.data); }

    __host__ __device__
    IndexType getIndex() const { return intr::atomic::load_acquire(&data) & INDEX_MASK; }

    __host__ __device__
    bool getFlag() const { return intr::atomic::load_acquire(&data) >> OFFSET; }

    __host__ __device__
    IndexType getRaw() const { return intr::atomic::load_acquire(&data); }
};

// node class
template <typename Type, typename IndexType, typename SizeType>
class Node{
    public:
        Type data;
        Link<IndexType> next;
        Link<IndexType> prev;
        int refcount;

        __host__ __device__
        Node()
            : next(AdrInfo<IndexType>::null(), false), prev(AdrInfo<IndexType>::null(), false) {}
}; // end of Node


// direct arena class
template <typename Type, typename IndexType, typename SizeType>
class DirectArena {
    public:
        Type arena[SizeType::VALUE];

        __host__ __device__
        Type& at(IndexType index)
        { return arena[index];}

        __host__ __device__
        const Type& at(IndexType index) const
        { return arena[index];}

        static constexpr size_t size()  
        { return SizeType::VALUE; }
}; // end of DirectArena

// simple slab arena
template <typename ARENA_SIZE, 
          template<typename> typename SLAB_PROXY_TYPE = defaultSlabProxy,
          typename SLAB_TYPE = Slab<DEFAULT_SLAB_SIZE>,
          typename SLAB_ADDR_TYPE = defaultSlabAddr>
class SlabArena {
    public: 
        typedef SLAB_ADDR_TYPE SlabAddrType;
        typedef SLAB_TYPE slabType;
        typedef SLAB_PROXY_TYPE<SLAB_TYPE> slabProxyType;
        typedef Node<slabProxyType, SlabAddrType, Size<2>> proxyNodeType;
        typedef Link<SlabAddrType> LinkType;

        static const size_t SLAB_COUNT = (ARENA_SIZE::VALUE + SLAB_TYPE::SIZE - 1) / SLAB_TYPE::SIZE;
        static const SlabAddrType NULL_IND = LinkType::INDEX_MASK;

        typedef DirectArena<slabType, SlabAddrType, Size<SLAB_COUNT>> BackingArenaType;
        typedef DirectArena<proxyNodeType, SlabAddrType, Size<SLAB_COUNT>> ProxyArenaType;

    private:
        BackingArenaType slabs;
        ProxyArenaType proxies;
        
        List<SlabAddrType> freeList;
        size_t nextSearch;

    public:
        __host__ __device__
        SlabArena() {
            intr::atomic::store_relaxed(&freeList.head, 0ull);
            intr::atomic::store_relaxed(&nextSearch, static_cast<size_t>(0));
            
            for(size_t i = 0; i < SLAB_COUNT; i++){
                intr::atomic::store_relaxed(&proxies.arena[i].data.allocState, static_cast<typename slabProxyType::AllocState>(0));
                intr::atomic::store_relaxed(&proxies.arena[i].data.allocMask, static_cast<typename slabProxyType::allocMaskElem>(0));
                intr::atomic::store_relaxed(&proxies.arena[i].data.reservationState, static_cast<uint32_t>(slabProxyType::FREE));

                // build list
                proxies.arena[i].next = (i == SLAB_COUNT - 1) ? LinkType{NULL_IND, false} : LinkType{i + 1, false};
                proxies.arena[i].prev = (i == 0) ? LinkType{NULL_IND, false} : LinkType{i - 1, false};
            }
        }


    private:
        __host__ __device__
        SlabAddrType deref(LinkType &linkAddr) {
            while (true) {
                if (linkAddr->getFlag() == true)
                    return NULL_IND;

                SlabAddrType nodeInd = linkAddr->getIndex();

                if (nodeInd == NULL_IND)
                    return NULL_IND;

                proxyNodeType& node = proxies.arena[nodeInd];
                intr::atomic::add_acq_rel(&node.refcount, 1);

                if (linkAddr->getIndex() == nodeInd)
                    return nodeInd;

                rel(linkAddr);
            }
        }

        __host__ __device__
        SlabAddrType deref_d(LinkType &linkAddr) {
            while (true) {
                SlabAddrType nodeInd = linkAddr->getIndex();

                if (nodeInd == NULL_IND)
                    return NULL_IND;

                proxyNodeType& node = proxies.arena[nodeInd];
                intr::atomic::add_acq_rel(&node.refcount, 1);

                if (linkAddr->getIndex() == nodeInd)
                    return nodeInd;

                rel(linkAddr);
            }
        }

    public:
        __host__ __device__
        SlabAddrType popList(SlabAddrType &head){
            while (true) {
                SlabAddrType currentInd = intr::atomic::load_acquire(&head);

                // Stop if the list is empty
                if (currentInd == NULL_IND)
                    return NULL_IND;

                proxyNodeType& current = proxies.arena[currentInd];
                LinkType nextLink = current.next;

                // Ensure deletion is completed and retry if the node is marked
                if (nextLink.getFlag() == true) {
                    helpDelete(head, currentInd);
                    continue;
                }

                // Set deletion mark of next link
                SlabAddrType nextLinkRaw = nextLink.getRaw();
                SlabAddrType old = intr::atomic::CAS_acq_rel(&current.next.data, nextLinkRaw,
                                                             LinkType::makeMarkedRaw(nextLinkRaw));
                if(old == nextLinkRaw){
                    // Set deletion mark of prev link
                    // and update next index of `prev`
                    helpDelete(head, currentInd);

                    // update prev index of `next`
                    SlabAddrType nextInd = current.next.getIndex();
                    if (nextInd != NULL_IND)
                        helpInsert(head, NULL_IND, nextInd);

                    return currentInd;
                }
            }
        } // end of pop

        __host__ __device__
        void pushList(SlabAddrType &head, SlabAddrType slabInd){
            if(slabInd >= SLAB_COUNT)
                return;

            proxyNodeType& node = proxies.arena[slabInd];
            node.prev = LinkType{NULL_IND, false};

            SlabAddrType nextInd = intr::atomic::load_acquire(&head);

            while (true) {
                node.next = LinkType{nextInd, false};

                SlabAddrType old = intr::atomic::CAS_acq_rel(&head, nextInd, slabInd);
                if (old == nextInd)
                    break;

                nextInd = old;
            };

            pushCommon(head, slabInd, nextInd);
        } // end of push

        __host__ __device__
        void pushCommon(SlabAddrType &head, SlabAddrType nodeInd, SlabAddrType nextInd) {
            proxyNodeType& node = proxies.arena[nodeInd];
            proxyNodeType& next = proxies.arena[nextInd];
            while (true) {
                LinkType link1 = next.prev;
                if (link1.getFlag() == true || node.next != LinkType{nextInd, false})
                    break;

                SlabAddrType link1Raw = link1.getRaw();
                SlabAddrType old = intr::atomic::CAS_acq_rel(&next.prev.data, link1Raw, nodeInd);
                if (old == link1Raw) {
                    if (node.prev.getFlag() == true)
                        helpInsert(head, nodeInd, nextInd);
                    break;
                }
            }
        }

        // Sets deletion mark of prev link
        // then updates next index of previous node
        __host__ __device__
        void helpDelete(SlabAddrType &head, SlabAddrType nodeInd) {
            // Set deletion mark of prev link
            markPrev(nodeInd);

            SlabAddrType lastInd = NULL_IND;
            proxyNodeType& node = proxies.arena[nodeInd];
            SlabAddrType prevInd = node.prev.getIndex();
            SlabAddrType nextInd = node.next.getIndex();

            // Loop to update next index of `prev`
            while (true) {
                proxyNodeType& next = proxies.arena[nextInd];

                // Ensure `next` is not marked
                if (next.next.getFlag() == true) {
                    markPrev(nextInd);
                    nextInd = next.next.getIndex();
                    continue;
                }

                // GARBAGE IF prevInd == NULL_IND
                proxyNodeType& prev = proxies.arena[prevInd];

                SlabAddrType prev2Ind;

                // Check if `node` is the head
                if (prevInd == NULL_IND)
                    prev2Ind = intr::atomic::load_acquire(&head);
                else {
                    LinkType prev2Link = prev.next;
                    prev2Ind = prev.next.getIndex();

                    // Ensure `prev` is not marked
                    if (prev2Link.getFlag() == true) {
                        if (lastInd != NULL_IND) {
                            markPrev(prevInd);

                            SlabAddrType next2Ind = prev.next.getIndex();

                            proxyNodeType& last = proxies.arena[lastInd];
                            intr::atomic::CAS_acq_rel(&last.next.data, prevInd, next2Ind);

                            prevInd = lastInd;
                            lastInd = NULL_IND;
                        } else {
                            prevInd = prev.prev.getIndex();
                        }
                        continue;
                    }
                }

                // Ensure `prev` is the previous node of `node`
                if (prev2Ind != nodeInd) {
                    lastInd = prevInd;
                    prevInd = prev2Ind;
                    continue;
                }

                // Update next index of `prev` with CAS
                SlabAddrType old;
                if (prevInd == NULL_IND)
                    old = intr::atomic::CAS_acq_rel(&head, nodeInd, nextInd);
                else
                    old = intr::atomic::CAS_acq_rel(&prev.next.data, nodeInd, nextInd);

                if (old == nodeInd)
                    break;
            }
        }

        // Updates then returns prev index of node
        __host__ __device__
        SlabAddrType helpInsert(SlabAddrType &head, SlabAddrType prevInd, SlabAddrType nodeInd) {
            proxyNodeType& node = proxies.arena[nodeInd];
            SlabAddrType lastInd = NULL_IND;

            // Loop to update prev index of `node`
            // prevInd == NULL_IND refers to what would be the head node
            while (true) {
                // A suggestion of a previous node
                // GARBAGE IF prevInd == NULL_IND
                proxyNodeType& prev = proxies.arena[prevInd];

                SlabAddrType prev2Ind;

                // Check if `node` is the head
                if (prevInd == NULL_IND) {
                    prev2Ind = intr::atomic::load_acquire(&head);
                } else {
                    LinkType prev2Link = prev.next;
                    prev2Ind = prev.next.getIndex();

                    // Ensure `prev` is not marked
                    if (prev2Link.getFlag() == true) {
                        if (lastInd != NULL_IND) {
                            markPrev(prevInd);

                            SlabAddrType next2Ind = prev.next.getIndex();

                            proxyNodeType& last = proxies.arena[lastInd];
                            intr::atomic::CAS_acq_rel(&last.next.data, prevInd, next2Ind);

                            prevInd = lastInd;
                            lastInd = NULL_IND;
                        } else {
                            prevInd = prev.prev.getIndex();
                        }
                        continue;
                    }
                }

                LinkType link1 = node.prev;

                // Abort if `node` is marked
                if (link1.getFlag() == true)
                    break;

                // Ensure `prev` is the previous node of `node`
                if (prev2Ind != nodeInd) {
                    lastInd = prevInd;
                    prevInd = prev2Ind;
                    continue;
                }

                // Abort if prev index of `node` is already updated
                if (link1.getIndex() == prevInd)
                    break;

                // Revalidate that `prev` is the predecessor
                bool prevIsPredecessor;
                if (prevInd == NULL_IND)
                    prevIsPredecessor = intr::atomic::load_acquire(&head) == nodeInd;
                else
                    prevIsPredecessor = prev.next.getRaw() == nodeInd;

                if (prevIsPredecessor) {
                    // Update prev index of `node` with CAS
                    SlabAddrType link1Raw = link1.getRaw();
                    SlabAddrType old = intr::atomic::CAS_acq_rel(&node.prev.data, link1Raw, prevInd);
                    // Check deletion mark of `prev.prev`
                    // to ensure `prev` was not deleted concurrently
                    // (not entirely necessary, but the benefit is well worth the extra check)
                    if (old == link1Raw && (prevInd == NULL_IND || prev.prev.getFlag() == false))
                        break;
                }
            }

            return prevInd;
        }

        __host__ __device__
        void markPrev(SlabAddrType nodeInd) {
            proxyNodeType& node = proxies.arena[nodeInd];
            while (true) {
                LinkType link1 = node.prev;

                if (link1.getFlag() == true)
                    break;

                SlabAddrType link1Raw = link1.getRaw();
                SlabAddrType old = intr::atomic::CAS_acq_rel(&node.prev.data, link1Raw,
                                                             LinkType::makeMarkedRaw(link1.getIndex()));
                if (old == link1Raw)
                    break;
            }
        }

        __host__ __device__
        void removeFromList(SlabAddrType &head, SlabAddrType slabInd){
            if(slabInd >= SLAB_COUNT)
                return;

            proxyNodeType& node = proxies.arena[slabInd];
            while (true) {
                LinkType link1 = node.next;
                if (link1.getFlag() == true)
                    break;
                SlabAddrType link1Raw = link1.getRaw();
                SlabAddrType old = intr::atomic::CAS_acq_rel(&node.next.data, link1Raw,
                                                             LinkType::makeMarkedRaw(link1.getIndex()));
                if (old == link1Raw) {
                    helpDelete(head, slabInd);

                    SlabAddrType prevInd = node.prev.getIndex();
                    SlabAddrType link1Ind = link1.getIndex();
                    if (link1Ind != NULL_IND)
                        helpInsert(head, prevInd, link1Ind);

                    break;
                }
            }
        } // end


        __host__ __device__
        SlabAddrType popFreeList(){
            SlabAddrType popped = popList(freeList.head);
            printf("Free list head is %llu after pop\n", freeList.head);
            return popped;
        } // end of pop

        __host__ __device__
        void pushFreeList(SlabAddrType slabInd){
            pushList(freeList.head,slabInd);
            printf("Free list head is %llu after push\n", freeList.head);
        } // end of push

        __host__ __device__
        void removeFromFreeList(SlabAddrType slabInd){
            removeFromList(freeList.head,slabInd);
        } // end

        __host__ __device__
        bool isValidPtr(void* ptr){
            if(!ptr)
                return false;
            
            SlabAddrType slabIndex = slabIndexFor(ptr);
            if(slabIndex >= SLAB_COUNT)
                return false;

            if(proxies.arena[slabIndex].data.getSize() == 0)
                return false;

            return true;
        } // end of isValidPtr

        __host__ __device__
        SlabAddrType slabIndexFor(void* ptr){
            char* bytePtr = static_cast<char*>(ptr);
            char* baseBytePtr = static_cast<char*>(static_cast<void*>(slabs.arena));
            
            size_t ptrOffset = bytePtr - baseBytePtr;
            size_t slabIndex = ptrOffset / SLAB_TYPE::SIZE;

            return static_cast<SlabAddrType>(slabIndex);
        } // end of slabIndexFor

        __host__ __device__
        slabType& slabAt(SlabAddrType slabIndex)
        { return slabs.arena[slabIndex];}

        __host__ __device__
        proxyNodeType& proxyAt(SlabAddrType slabIndex)
        { return proxies.arena[slabIndex];}

        __host__ __device__
        slabType& slabFor(void* ptr){
            SlabAddrType slabIndex = slabIndexFor(ptr);
            return slabs.arena[slabIndex];
        }

        __host__ __device__
        proxyNodeType& proxyFor(void* ptr){
            SlabAddrType slabIndex = slabIndexFor(ptr);
            return proxies.arena[slabIndex];
        }


        __host__ __device__
        SlabAddrType alloc(size_t objectSize){
            if(objectSize == 0) {
                objectSize = 1;
            }

            // try free slab from the free list
            SlabAddrType freeSlab = popFreeList();

            while (freeSlab != NULL_IND) {
                slabProxyType& proxy = proxies.arena[freeSlab].data;
                slabType& slab = slabs.arena[freeSlab];
                if (proxy.tryClaim(&slab, objectSize)) {
                    return freeSlab;
                } else {
                    freeSlab = popFreeList();
                }
            }

            printf("No free slabs in free list\n");
            
            return NULL_IND;
        } // end of alloc
        

        __host__ __device__
        bool free(SlabAddrType slabAddr){
            if(slabAddr >= SLAB_COUNT)
                return false;
            bool success = proxies.arena[slabAddr].data.clearAllocState();
            if (success)
                pushFreeList(slabAddr);
            
            return success;
        } // end of free

        __host__ __device__
        void getStats(size_t& totalSlabs, size_t& usedSlabs, size_t& emptyReusables, size_t& freeSlabs){
            totalSlabs = SLAB_COUNT;
            usedSlabs = 0;
            emptyReusables = 0;
            freeSlabs = 0;

            SlabAddrType current = intr::atomic::load_acquire(&freeList.head);
            while (current != NULL_IND) {
                freeSlabs++;
                current = proxies.arena[current].next;
            }

            for(size_t i = 0; i < SLAB_COUNT; i++){
                slabProxyType& proxy = proxies.arena[i].data;
                if(proxy.getSize() > 0){
                    if(proxy.isEmpty()){
                        emptyReusables++;
                    } else {
                        usedSlabs++;
                    }
                }
            }
        } // end of getStats

        __host__ __device__
        bool validateFreeList(SlabAddrType &head) {
            size_t count = 0;
            SlabAddrType currentInd = intr::atomic::load_acquire(&head);
            SlabAddrType prevInd = NULL_IND;
            
            while (currentInd != NULL_IND && count < SLAB_COUNT) {
                proxyNodeType& current = proxies.arena[currentInd];

                // check prev mark
                if (current.prev.getFlag() == true) {
                    printf("INVALID: prev is marked\n");
                    return false;
                }

                // check next mark
                if (current.next.getFlag() == true) {
                    printf("INVALID: next is marked\n");
                    return false;
                }

                // check that prev
                if (current.prev.getIndex() != prevInd) {
                    printf("INVALID: prev is not updated\n");
                    return false;
                }

                // check slab is free
                if (current.data.getSize() != 0) {
                    printf("INVALID: slab isn't free\n");
                    return false;
                }

                prevInd = currentInd;
                currentInd = current.next.getIndex();
                count++;
            }

            return count < SLAB_COUNT;
        }

        __host__ __device__
        bool validatePartialList(SlabAddrType &head) {
            size_t count = 0;
            SlabAddrType currentInd = intr::atomic::load_acquire(&head);
            SlabAddrType prevInd = NULL_IND;
            
            while (currentInd != NULL_IND && count < SLAB_COUNT) {
                proxyNodeType& current = proxies.arena[currentInd];

                // check prev mark
                if (current.prev.getFlag() == true) {
                    printf("INVALID: prev is marked\n");
                    return false;
                }

                // check next mark
                if (current.next.getFlag() == true) {
                    printf("INVALID: next is marked\n");
                    return false;
                }

                // check that prev
                if (current.prev.getIndex() != prevInd) {
                    printf("INVALID: prev is not updated\n");
                    return false;
                }

                // check slab is partial
                if (current.data.isAvailable() == 0) {
                    printf("INVALID: slab is partial\n");
                    return false;
                }

                prevInd = currentInd;
                currentInd = current.next.getIndex();
                count++;
            }

            return count < SLAB_COUNT;
        }
}; // end of SlabArena

// test allocator
template <typename SLAB_ALLOCATOR_TYPE>
class SimpleAllocator {
    public:
        typedef SLAB_ALLOCATOR_TYPE SlabAllocatorType;
        typedef typename SlabAllocatorType::slabType SlabType;
        typedef typename SlabAllocatorType::SlabAddrType SlabAddrType;
        typedef typename SlabAllocatorType::slabProxyType SlabProxyType;
        static const SlabAddrType NULL_IND = SlabAllocatorType::NULL_IND;
        static const size_t SLAB_COUNT = SlabAllocatorType::SLAB_COUNT;

    private:
        SlabAllocatorType& slabAllocator;
        size_t objectSize;
        List<SlabAddrType> partialList;


        __host__ __device__
        SlabAddrType popPartialList(){
            return slabAllocator.popList(partialList.head);
        } // end of pop

        __host__ __device__
        void pushPartialList(SlabAddrType slabInd){
            slabAllocator.pushList(partialList.head,slabInd);
            slabAllocator.validatePartialList(partialList.head);
            printf("Partial list head is %llu after push\n", partialList.head);
        } // end of push

        __host__ __device__
        void removeFromPartialList(SlabAddrType slabInd){
            slabAllocator.removeFromList(partialList.head,slabInd);
            slabAllocator.validatePartialList(partialList.head);
            printf("Partial list head is %llu after remove\n", partialList.head);
        } // end


    public:
        __host__ __device__
        SimpleAllocator(SlabAllocatorType& slabAlloc, size_t objSize)
            : slabAllocator(slabAlloc)
            , objectSize(objSize == 0 ? 1 : objSize)
            , partialList{NULL_IND}
        {}




        __host__ __device__
        bool isValidPtr(void* ptr) {
            return slabAllocator.isValidPtr(ptr);
        }

        __host__ __device__
        size_t getAllocatedCount() {
            size_t totalCount = 0;
            
            // Iterate through all slabs in the allocator
            for(size_t i = 0; i < SlabAllocatorType::SLAB_COUNT; i++) {
                SlabProxyType& slabProxy = slabAllocator.proxyAt(static_cast<SlabAddrType>(i)).data;
                
                // Only count slabs that are allocated for this object size
                if(slabProxy.getSize() == objectSize) {
                    totalCount += (slabProxy.allocState & SlabProxyType::COUNT_MASK);
                }
            }
            
            return totalCount;
        } // end of getCount

        __host__ __device__
        void getReuseStats(size_t& activeSlabs, size_t& reusableSlabs, size_t& totalObjects){
            activeSlabs = 0;
            reusableSlabs = 0;
            totalObjects = 0;

            for(size_t i = 0; i < SlabAllocatorType::SLAB_COUNT; i++){
                SlabProxyType& slabProxy = slabAllocator.proxyAt(static_cast<SlabAddrType>(i)).data;

                if(slabProxy.getSize() == objectSize){
                    size_t count = slabProxy.getCount();
                    totalObjects += count;

                    if(count > 0){
                        activeSlabs++;
                    } else {
                        reusableSlabs++;
                    } // else
                } // outer if
            } // for
        } // end of getReuse

        __host__ __device__
        void* allocObject(SlabAddrType slabAddr, bool& slabFilled){
            SlabType& slab = slabAllocator.slabAt(slabAddr);
            SlabProxyType& slabProxy = slabAllocator.proxyAt(slabAddr).data;

            void* result = slabProxy.alloc(&slab, slabFilled);

            return result;
        }

        __host__ __device__
        void* alloc(){
            SlabAddrType slabAddr = intr::atomic::load_acquire(&partialList.head);
            bool slabFilled = false;
            void* result = nullptr;

            // Check if partial list is empty
            // Loop if other thread filled slab at partialList.head
            // before this one could
            while (slabAddr != NULL_IND && result == nullptr) {
                printf("Partial slab found\n");
                result = allocObject(slabAddr, slabFilled);
                slabAddr = intr::atomic::load_acquire(&partialList.head);
            }

            // Use free list if partial list is empty
            if (result == nullptr) {
                slabAddr = slabAllocator.alloc(objectSize);
                if(slabAddr == NULL_IND)
                    return nullptr;
                pushPartialList(slabAddr);
                result = allocObject(slabAddr, slabFilled);
            }

            if (slabFilled)
                removeFromPartialList(slabAddr);

            return result;
        } // end of alloc

        __host__ __device__
        bool free(void* ptr){
            if (!isValidPtr(ptr))
                return false;

            SlabAddrType slabAddr = slabAllocator.slabIndexFor(ptr);
            SlabType& slab = slabAllocator.slabAt(slabAddr);
            SlabProxyType& slabProxy = slabAllocator.proxyAt(slabAddr).data;

            bool slabEmptied = false;
            bool slabWasFull = false;
            if(!slabProxy.free(&slab, ptr, slabEmptied, slabWasFull))
                return false;

            if (slabEmptied) {
                printf("Slab emptied\n");
                removeFromPartialList(slabAddr);
                slabAllocator.free(slabAddr);
            } else if (slabWasFull) {
                pushPartialList(slabAddr);
            }

            return true;
        } // end of free
}; // end of simpleAlloc



// smaller slabs!!
typedef SlabArena<Size<1024*1024>, defaultSlabProxy, Slab<4096>> TestSlabArena; // 4KB slabs
typedef SimpleAllocator<TestSlabArena> TestAllocator;
