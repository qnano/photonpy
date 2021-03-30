#pragma once

#include <cstdint>
#include <limits>
#include "conditional.h"

// Based on:
// http://stackoverflow.com/a/5970314/383299
// https://gist.github.com/ashelly/5665911
// https://github.com/craffel/median-filter
// https://github.com/suomela/median-filter
//
// Sliding median filter
// Portions Copyright (c) 2018 sergiy.yevtushenko(at)gmail.com under <http://www.opensource.org/licenses/mit-license>
// Created 2012 by Colin Raffel
// Portions Copyright (c) 2011 ashelly.myopenid.com under <http://www.opensource.org/licenses/mit-license>

// Modified to use dynamic window size by Jelmer Cnossen

namespace siy {
    template<typename ValueType>
    class median_filter {

            using IndexType = int;

            std::vector<ValueType> data;
            std::vector<IndexType> pos, heapStorage;

            //ValueType data[N]; // Circular queue of values
            //IndexType pos[N];   // Index into `heap` for each value
            //IndexType heapStorage[N]; // heap holds a pointer to the middle of its data; this is where the data is allocated.
            IndexType *heap;  // Max/median/min heap holding indexes into `data`.
            IndexType idx = 0;    // Position in circular queue
            IndexType minCount = 0;  // Count of items in min heap
            IndexType maxCount = 0;  // Count of items in max heap
            int N;

        public:
            median_filter(int N) : N(N), data(N), pos(N), heapStorage(N) {
                heap = &heapStorage[N / 2];

                IndexType nItems = N;
                // Set up initial heap fill pattern: median, max, min, max, ...
                while (nItems--) {
                    pos[nItems] = ((nItems + 1) / 2) * ((nItems & 1) ? -1 : 1);
                    heap[pos[nItems]] = nItems;
                }
            }

            // Inserts item, maintains median in O(lg nItems)
            ValueType filter(ValueType v) {
                IndexType p = pos[idx];
                ValueType old = data[idx];
                data[idx] = v;

                idx = (idx + 1) % N;

                if (p > 0) { // New item is in minheap
                    return itemInMinHeap(v, p, old);
                } else if (p < 0) {   // New item is in maxheap
                    return itemInMaxHeap(v, p, old);
                } else { // New item is at median
                    return itemAtMedian();
                }

                return last();
            }

            // Returns median item (or average of 2 when item count is even)
            ValueType last() {
                ValueType v = data[heap[0]];
                if (minCount < maxCount) {
                    v = (v + data[heap[-1]]) / 2;
                }
                return v;
            }

        private:
            ValueType itemAtMedian() {
                if (maxCount && maxSortUp(-1)) {
                    maxSortDown(-1);
                }
                if (minCount && minSortUp(1)) {
                    minSortDown(1);
                }
                return last();
            }

            ValueType itemInMaxHeap(ValueType v, IndexType p, ValueType old) {
                if (maxCount < N / 2) {
                    maxCount++;
                } else if (v < old) {
                    maxSortDown(p);
                    return last();
                }
                if (maxSortUp(p) && minCount && mmCmpExch(1, 0)) {
                    minSortDown(1);
                }
                return last();
            }

            ValueType itemInMinHeap(ValueType v, IndexType p, ValueType old) {
                if (minCount < (N - 1) / 2) {
                    minCount++;
                } else if (v > old) {
                    minSortDown(p);
                    return last();
                }
                if (minSortUp(p) && mmCmpExch(0, -1)) {
                    maxSortDown(-1);
                }
                return last();
            }

            // Swaps items i&j in heap, maintains indexes
            bool mmexchange(IndexType i, IndexType j) {
                auto t = heap[i];
                heap[i] = heap[j];
                heap[j] = t;
                pos[heap[i]] = i;
                pos[heap[j]] = j;
                return true;
            }

            // Maintains minheap property for all items below i.
            void minSortDown(IndexType i) {
                for (i *= 2; i <= minCount; i *= 2) {
                    if (i < minCount && mmless(i + 1, i)) {
                        ++i;
                    }
                    if (!mmCmpExch(i, i / 2)) {
                        break;
                    }
                }
            }

            // Maintains maxheap property for all items below i. (negative indexes)
            void maxSortDown(IndexType i) {
                for (i *= 2; i >= -maxCount; i *= 2) {
                    if (i > -maxCount && mmless(i, i - 1)) {
                        --i;
                    }
                    if (!mmCmpExch(i / 2, i)) {
                        break;
                    }
                }
            }

            // Returns 1 if heap[i] < heap[j]
            bool mmless(IndexType i, IndexType j) {
                return (data[heap[i]] < data[heap[j]]);
            }

            // Swaps items i&j if i<j; returns true if swapped
            bool mmCmpExch(IndexType i, IndexType j) {
                return (mmless(i, j) && mmexchange(i, j));
            }

            // Maintains minheap property for all items above i, including median
            // Returns true if median changed
            bool minSortUp(IndexType i) {
                while (i > 0 && mmCmpExch(i, i / 2)) {
                    i /= 2;
                }
                return (i == 0);
            }

            // Maintains maxheap property for all items above i, including median
            // Returns true if median changed
            bool maxSortUp(IndexType i) {
                while (i < 0 && mmCmpExch(i / 2, i)) {
                    i /= 2;
                }
                return (i == 0);
            }
    };
}   // namespace
