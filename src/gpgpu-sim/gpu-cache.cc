// Copyright (c) 2009-2011, Tor M. Aamodt, Tayler Hetherington
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "gpu-cache.h"
#include "stat-tool.h"
#include <assert.h>

#include "../cuda-sim/cuda-sim.h"
#include <iostream>

using namespace std;

#define MAX_DEFAULT_CACHE_SIZE_MULTIBLIER 4
// used to allocate memory that is large enough to adapt the changes in cache size across kernels
//#define COLLECT_INNER_USE
//#define PRINT_TRACE
#ifdef PRINT_TRACE
#include <fstream>
ofstream trace_file;
#endif

const char * cache_request_status_str(enum cache_request_status status) 
{
   static const char * static_cache_request_status_str[] = {
      "HIT",
      "HIT_RESERVED",
      "MISS",
      "MISS_PARTIAL",
      "RESERVATION_FAIL"
   }; 

   assert(sizeof(static_cache_request_status_str) / sizeof(const char*) == NUM_CACHE_REQUEST_STATUS); 
   assert(status < NUM_CACHE_REQUEST_STATUS); 

   return static_cache_request_status_str[status]; 
}

unsigned l1d_cache_config::set_index(new_addr_type addr) const{
    unsigned set_index = m_nset; // Default to linear set index function
    unsigned lower_xor = 0;
    unsigned upper_xor = 0;

    switch(m_set_index_function){
    case FERMI_HASH_SET_FUNCTION:
        /*
        * Set Indexing function from "A Detailed GPU Cache Model Based on Reuse Distance Theory"
        * Cedric Nugteren et al.
        * ISCA 2014
        */
        if(m_nset == 32 || m_nset == 64){
            // Lower xor value is bits 7-11
            lower_xor = (addr >> m_line_sz_log2) & 0x1F;

            // Upper xor value is bits 13, 14, 15, 17, and 19
            //upper_xor  = (addr & 0xE000)  >> 13; // Bits 13, 14, 15
            //upper_xor |= (addr & 0x20000) >> 14; // Bit 17
            //upper_xor |= (addr & 0x80000) >> 15; // Bit 19
            // lld: for other line size
            upper_xor  = (addr >> (m_line_sz_log2 + 6)) & 0x7; // Bits 13, 14, 15
            upper_xor |= (addr >> (m_line_sz_log2 + 7)) & 0x8; // Bit 17
            upper_xor |= (addr >> (m_line_sz_log2 + 8)) & 0x10; // Bit 19

            set_index = (lower_xor ^ upper_xor);

            // 48KB cache prepends the set_index with bit 12
            if(m_nset == 64)
                //set_index |= (addr & 0x1000) >> 7;
                set_index |= (addr >> m_line_sz_log2) & 0x20; // lld: for other line size

        }else{ /* Else incorrect number of sets for the hashing function */
            assert("\nGPGPU-Sim cache configuration error: The number of sets should be "
                    "32 or 64 for the hashing set index function.\n" && 0);
        }
        break;

    case CUSTOM_SET_FUNCTION:
        /* No custom set function implemented */
        break;

    case LINEAR_SET_FUNCTION:
        set_index = (addr >> m_line_sz_log2) & (m_nset-1);
        break;
    }

    // Linear function selected or custom set index function not implemented
    assert((set_index < m_nset) && "\nError: Set index out of bounds. This is caused by "
            "an incorrect or unimplemented custom set index function.\n");

    return set_index;
}

void l2_cache_config::init(linear_to_raw_address_translation *address_mapping){
	cache_config::init(m_config_string,FuncCachePreferNone);
	m_address_mapping = address_mapping;
}

unsigned l2_cache_config::set_index(new_addr_type addr) const{
	if(!m_address_mapping){
		return(addr >> m_line_sz_log2) & (m_nset-1);
	}else{
		// Calculate set index without memory partition bits to reduce set camping
		new_addr_type part_addr = m_address_mapping->partition_address(addr);
		return(part_addr >> m_line_sz_log2) & (m_nset -1);
	}
}

tag_array::~tag_array() 
{
    delete[] m_lines;
    delete[] m_sets; // lld
}

tag_array::tag_array( cache_config &config,
                      int core_id,
                      int type_id,
                      cache_block_t* new_lines)
    : m_config( config ),
      m_lines( new_lines )
{
    init( core_id, type_id );
}

void tag_array::update_cache_parameters(cache_config &config)
{
	m_config=config;
}

tag_array::tag_array( cache_config &config,
                      int core_id,
                      int type_id )
    : m_config( config )
{
    //assert( m_config.m_write_policy == READ_ONLY ); Old assert
    m_lines = new cache_block_t[MAX_DEFAULT_CACHE_SIZE_MULTIBLIER*config.get_num_lines()];
    init( core_id, type_id );
}

void tag_array::init( int core_id, int type_id )
{
    m_access = 0;
    m_miss = 0;
    m_pending_hit = 0;
    m_res_fail = 0;
    // initialize snapshot counters for visualizer
    m_prev_snapshot_access = 0;
    m_prev_snapshot_miss = 0;
    m_prev_snapshot_pending_hit = 0;
    m_core_id = core_id; 
    m_type_id = type_id;

    // lld
    m_fill = 0;
    m_total_lat = 0;
    m_hit_fill = 0;
    m_total_hit_lat = 0;
    for(unsigned i = 0; i <= MAX_MEMORY_ACCESS_SIZE; i++)
        m_reuse[i] = 0;
    for(unsigned i = 0; i <= MAX_CACHE_CHUNK_NUM; i++) {
        m_chunk_reuse[i] = 0;
        m_chunk_cont_reuse[i] = 0;
    }

    if(m_config.m_replacement_policy == ADAP_GRAN) { // lld: initialize set and block status
        m_sets = new cache_set_t[m_config.m_nset];
        for(unsigned i=0; i < m_config.m_nset; i++)
            m_sets[i].init(m_config.m_assoc);
        for (unsigned i=0; i < m_config.get_num_lines(); i++) {
            if(i % m_config.m_assoc >= m_config.m_true_assoc)
                m_lines[i].m_status = UNUSED;
        }
    }

    if( m_config.m_custom_repl == 1 )
        init_repl(); // lld: initialize replacement status

#ifdef PRINT_TRACE
    if(m_core_id == 0 && m_type_id == 0) {
        trace_file.open("/tmp/trace.txt", ofstream::out);
    }
#endif
}

enum cache_request_status tag_array::probe( mem_fetch *mf, new_addr_type addr, unsigned &idx ) {
    //assert( m_config.m_write_policy == READ_ONLY );
    unsigned set_index = m_config.set_index(addr);
    new_addr_type tag = m_config.tag(addr);

    unsigned invalid_line = (unsigned)-1;
    /*
    unsigned valid_line = (unsigned)-1;
    unsigned valid_timestamp = (unsigned)-1;
    */

    bool all_reserved = true;
    bool all_used = true; // lld
    bool have_valid = false; // lld
    idx = (unsigned)-1; // lld

    // check for hit or pending hit
    cache_request_status status = MISS;
    for (unsigned way=0; way<m_config.m_assoc; way++) {
        unsigned index = set_index*m_config.m_assoc+way;
        cache_block_t *line = &m_lines[index];
        if (line->m_tag == tag && line->m_status != INVALID && line->m_status != UNUSED && status == MISS) { // lld
            // lld: sector cache
            if( m_config.m_replacement_policy == SECTOR || m_config.m_replacement_policy == ADAP_GRAN )
            {
                idx = index;
                if ( line->m_status == RESERVED )
                    status = HIT_RESERVED;
                else if ( line->m_status == VALID )
                    status = HIT;
                else if ( line->m_status == MODIFIED )
                    status = HIT;
                else
                    assert(0);

                for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++)
                    if(mf->get_original_sector_mask(m_core_id<0).test(i) && !line->m_sector_mask.test(i)) // lld: sector and adaptive grain cache
                        status = MISS_PARTIAL;
                if(status != MISS_PARTIAL) {
                    for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++)
                        if(mf->get_original_sector_mask(m_core_id<0).test(i) && !line->m_ready_sector_mask.test(i)) // lld: sector and adaptive grain cache
                            return HIT_RESERVED;
                    return HIT;
                }
            }
            else
            {
                if ( line->m_status == RESERVED ) {
                    idx = index;
                    return HIT_RESERVED;
                } else if ( line->m_status == VALID ) {
                    idx = index;
                    return HIT;
                } else if ( line->m_status == MODIFIED ) {
                    idx = index;
                    return HIT;
                } else {
                    assert( line->m_status == INVALID );
                }
            }
        }
        if (line->m_status != RESERVED && line->m_status != UNUSED && index != idx) { // lld: cannot replace itself
            all_reserved = false;
            if (line->m_status == INVALID) {
                invalid_line = index;
            /*
            } else {
                // valid line : keep track of most appropriate replacement candidate
                if ( m_config.m_replacement_policy == LRU ) {
                    if ( line->m_last_access_time < valid_timestamp ) {
                        valid_timestamp = line->m_last_access_time;
                        valid_line = index;
                    }
                } else if ( m_config.m_replacement_policy == FIFO ) {
                    if ( line->m_alloc_time < valid_timestamp ) {
                        valid_timestamp = line->m_alloc_time;
                        valid_line = index;
                    }
                }
            */
            }
        }
        if(line->m_status == UNUSED) // lld
            all_used = false;
        if((line->m_status == VALID || line->m_status == MODIFIED) && index != idx) // lld
            have_valid = true;
    }

    /*
    if( m_config.m_replacement_policy == ADAP_GRAN && m_config.m_custom_repl == 1 ) {
        if(mf->get_first_touch(m_core_id<0) && mf->get_done(m_core_id<0)) {
            decide_allocation(mf, addr); // lld: decide allocation
            decide_prefetch(mf, addr); // lld: decide prefetch
        }
    }
    */

    // lld: potentially replace on MISS_PARTIAL
    update_sets(set_index);
    if(m_config.m_replacement_policy == ADAP_GRAN && status == MISS_PARTIAL) {
        unsigned extra_size = 0;
        for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++)
            if(mf->get_alloc_sector_mask(m_core_id<0).test(i) && !m_lines[idx].m_sector_mask.test(i))
                extra_size++;

        if(m_sets[set_index].m_size + extra_size <= m_config.m_assoc)
            return status;

        if( m_config.m_custom_repl == 1 && mf->get_first_touch(m_core_id<0) && mf->get_done(m_core_id<0) ) {
            decide_allocation(mf, addr); // lld: decide allocation
            decide_prefetch(mf, addr); // lld: decide prefetch
        }
        /*
        */
        extra_size = 0;
        for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++)
            if(mf->get_alloc_sector_mask(m_core_id<0).test(i) && !m_lines[idx].m_sector_mask.test(i))
                extra_size++;
        if(m_sets[set_index].m_size + extra_size <= m_config.m_assoc)
            return status;

        if ( !have_valid ) {
            assert( m_config.m_alloc_policy == ON_MISS ); 
            return RESERVATION_FAIL; // miss and not enough space in cache to allocate on miss
        }

        return status;
    } else if(m_config.m_replacement_policy == SECTOR && status == MISS_PARTIAL) {
        if( m_config.m_custom_repl == 1 && mf->get_first_touch(m_core_id<0) && mf->get_done(m_core_id<0) ) {
            decide_prefetch(mf, addr); // lld: decide prefetch
        }
        return status;
    }

    if( m_config.m_replacement_policy == ADAP_GRAN ) {
        assert(status == MISS);
        if(m_sets[set_index].m_size + mf->get_alloc_sector_mask(m_core_id<0).count() <= m_config.m_assoc) {
            assert(mf->get_alloc_sector_mask(m_core_id<0).count());
            if ( invalid_line != (unsigned)-1 )
                return MISS;
            else if(!all_used)
                return MISS;
        }

        if( m_config.m_custom_repl == 1 && mf->get_first_touch(m_core_id<0) && mf->get_done(m_core_id<0) ) {
            decide_allocation(mf, addr); // lld: decide allocation
            decide_prefetch(mf, addr); // lld: decide prefetch
        }
        /*
        */

        if(m_sets[set_index].m_size + mf->get_alloc_sector_mask(m_core_id<0).count() > m_config.m_assoc) {
            if(!have_valid)
                return RESERVATION_FAIL;
        } else if(mf->get_alloc_sector_mask(m_core_id<0).count() == 0) // lld: equal to bypass
            return MISS;
        else if(invalid_line != (unsigned)-1)
            return MISS;
        else if(!all_used)
            return MISS;
    }
    if ( all_reserved ) {
        assert( m_config.m_alloc_policy == ON_MISS ); 
        return RESERVATION_FAIL; // miss and not enough space in cache to allocate on miss
    }
    if( m_config.m_replacement_policy == PARTITION && !check_avail(mf, addr) ) // lld: check reservation fail for partition
        return RESERVATION_FAIL;

    /*
    if ( invalid_line != (unsigned)-1 ) {
        idx = invalid_line;
    } else if ( valid_line != (unsigned)-1) {
        idx = valid_line;
    } else abort(); // if an unreserved block exists, it is either invalid or replaceable 
    */

    return MISS;
}

enum cache_request_status tag_array::access( mem_fetch *mf, new_addr_type addr, unsigned time, unsigned &idx )
{
    bool wb=false;
    cache_block_t evicted;
    enum cache_request_status result = access(mf,addr,time,idx,wb,evicted);
    assert(!wb);
    return result;
}

enum cache_request_status tag_array::access( mem_fetch *mf, new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted ) 
{
    m_access++;
    shader_cache_access_log(m_core_id, m_type_id, 0); // log accesses to cache
    list<unsigned> repl_set;
    if(m_config.m_custom_repl == 1 && m_config.m_replacement_policy == ADAP_GRAN && mf->get_done(m_core_id<0)) // lld: touch on first access
        touch(mf, addr);
    enum cache_request_status status = probe_selector(mf,addr,idx,repl_set,time);
    switch (status) {
    case HIT_RESERVED: 
        m_pending_hit++;
    case HIT: 
        if(!mf->get_prefetch(m_core_id<0)) { // lld
            m_lines[idx].m_last_access_time=time; 
            // lld: update statistics
            update_stat(mf, idx, true);
        }
        if ( m_config.m_custom_repl == 1 ) // lld: update on hit
            update(mf, addr, idx, status);
        break;
    case MISS_PARTIAL: // lld: sector cache
        assert( idx != (unsigned)-1 );
        if(!mf->get_prefetch(m_core_id<0)) // lld
            m_lines[idx].m_last_access_time=time; 
        if ( m_config.m_alloc_policy == ON_MISS ) { // lld: sector and adaptive grain cache
            if(m_config.m_replacement_policy == ADAP_GRAN) {
                // replace
                if(!repl_set.empty()) {
                    m_partial_miss_repl++;
                    unsigned repl_idx = repl_set.front();
                    if(repl_idx == (unsigned)-1) {
                        assert(mf->get_done(m_core_id<0));
                        break;
                    }
                    if( m_lines[repl_idx].m_status == MODIFIED ) {
                        wb = true;
                        evicted = m_lines[repl_idx];
                    }
                    update_stat(mf, repl_idx, false);
                    m_lines[repl_idx].m_status = INVALID;
                }

                // check if there is enough replacement
                unsigned set_index = m_config.set_index(addr);
                update_sets(set_index);
                unsigned extra_size = 0;
                for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++)
                    if(mf->get_alloc_sector_mask(m_core_id<0).test(i) && !m_lines[idx].m_sector_mask.test(i))
                        extra_size++;
                if(m_sets[set_index].m_size + extra_size > m_config.m_assoc) { // need more replacement
                    mf->set_done(m_core_id<0, false);
                    break;
                } else
                    mf->set_done(m_core_id<0, true);
            }

            if(!mf->get_prefetch(m_core_id<0)) { // lld
                m_lines[idx].m_last_access_time=time; 
                update_stat(mf, idx, true);
            } else {
                assert(m_config.m_replacement_policy == PREF);
                m_issued_prefetch++;
            }

            // update sector mask of cache line
            for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++)
                if(mf->get_alloc_sector_mask(m_core_id<0).test(i) && !m_lines[idx].m_sector_mask.test(i)) {
                    assert(!m_lines[idx].m_original_sector_mask.test(i));
                    m_lines[idx].m_sector_mask.set(i);
                    if(mf->get_original_sector_mask(m_core_id<0).test(i)) {
                        m_lines[idx].m_original_sector_mask.set(i);
                        m_desired_prefetch++;
                    } else {
                        assert(!m_lines[idx].m_pref_sector_mask.test(i));
                        m_lines[idx].m_pref_sector_mask.set(i);
                    }
                }
            if ( m_config.m_custom_repl == 1 ) // lld: update on partial miss
                update(mf, addr, idx, status);
        }
        break;
    case MISS:
        m_miss++;
        shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
        if ( m_config.m_alloc_policy == ON_MISS ) {
            if ( idx != (unsigned)-1 ) { // lld: -1 means bypass
                m_miss_repl++;
                if( m_lines[idx].m_status == MODIFIED ) {
                    wb = true;
                    evicted = m_lines[idx];
                }
                update_stat(mf, idx, false);

                // lld: check if need multiple replacement
                if(m_config.m_replacement_policy == ADAP_GRAN) {
                    m_lines[idx].m_status = INVALID;
                    unsigned set_index = m_config.set_index(addr);
                    update_sets(set_index);
                    if(m_sets[set_index].m_size + mf->get_alloc_sector_mask(m_core_id<0).count() > m_config.m_assoc) { // need more replacement
                        mf->set_done(m_core_id<0, false);
                        break;
                    } else
                        mf->set_done(m_core_id<0, true);
                }

                m_lines[idx].allocate( m_config.tag(addr), m_config.block_addr(addr), time, mf->get_wid() );
#ifdef COLLECT_INNER_USE
                m_lines[idx].m_reuse = mf->get_access_byte_mask(); // lld: for use information
#endif
                for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++) // lld: sector and adaptive grain cache
                    if(mf->get_alloc_sector_mask(m_core_id<0).test(i)) {
                        m_lines[idx].m_sector_mask.set(i);
                        if(mf->get_original_sector_mask(m_core_id<0).test(i))
                            m_lines[idx].m_original_sector_mask.set(i);
                        else
                            m_lines[idx].m_pref_sector_mask.set(i);
                    }
                if(mf->get_prefetch(m_core_id<0)) { // lld
                    assert(m_config.m_replacement_policy == PREF);
                    m_lines[idx].m_prefetch = true;
                    m_issued_prefetch++;
                }
                if ( m_config.m_custom_repl == 1 ) // lld: update on miss
                    update(mf, addr, idx, status);
            } else
                assert(mf->get_done(m_core_id<0));
        }
        break;
    case RESERVATION_FAIL:
        assert(0); // lld
        m_res_fail++;
        shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
        break;
    default:
        fprintf( stderr, "tag_array::access - Error: Unknown"
            "cache_request_status %d\n", status );
        abort();
    }
    mf->set_access_status((m_core_id<0), status); // lld
    return status;
}

unsigned tag_array::fill( mem_fetch *mf, new_addr_type addr, unsigned time )
{
    assert( m_config.m_alloc_policy == ON_FILL );
    unsigned idx;

    // lld: if allocate on fill, this probe will look for the victim
    //enum cache_request_status status = probe(addr,idx);
    list<unsigned> repl_set;
    enum cache_request_status status = access_probe(mf,addr,idx,repl_set,time);
    // FIXME: multiple replacement is not implemented for ON_FILL
    assert(status==MISS || status==MISS_PARTIAL); // MSHR should have prevented redundant memory request. lld: sector cache
    if ( idx != (unsigned)-1 ) { // lld: -1 means bypass
        if(status==MISS_PARTIAL) {
            update_stat(mf, idx, true);
            if(mf->get_is_write())
                m_lines[idx].m_status = MODIFIED;
            for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++)
                if(mf->get_sector_mask(m_core_id<0).test(i)) {
                    assert(!m_lines[idx].m_sector_mask.test(i)); // lld: sector and adaptive grain cache
                    m_lines[idx].m_sector_mask.set(i);
                    m_lines[idx].m_ready_sector_mask.set(i); // lld: sector and adaptive grain cache
                    if(mf->get_original_sector_mask(m_core_id<0).test(i))
                        m_lines[idx].m_original_sector_mask.set(i);
                }
        } else {
            update_stat(mf, idx, false);
            m_lines[idx].allocate( m_config.tag(addr), m_config.block_addr(addr), time, mf->get_wid() );
#ifdef COLLECT_INNER_USE
            m_lines[idx].m_reuse = mf->get_access_byte_mask(); // lld: for use information
#endif
            if(mf->get_prefetch(m_core_id<0))
                m_lines[idx].m_prefetch = true;
            m_lines[idx].fill(time, mf->get_wid());
            for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++) // lld: sector and adaptive grain cache
                if(mf->get_sector_mask(m_core_id<0).test(i)) {
                    m_lines[idx].m_sector_mask.set(i);
                    m_lines[idx].m_ready_sector_mask.set(i);
                    if(mf->get_original_sector_mask(m_core_id<0).test(i))
                        m_lines[idx].m_original_sector_mask.set(i);
                }
        }

        if ( m_config.m_custom_repl == 1 ) // lld: update on miss
            update(mf, addr, idx, status);
    }
    // lld: for memory latency calculation
    if( m_type_id != 1 && mf->get_issue(m_core_id<0) )
    {
        mf->set_ret_time(m_core_id<0, time);
        assert(mf->get_issue_time(m_core_id<0));
        if(m_core_id >= 0 && (mf->get_status(true) == HIT || mf->get_status(true) == HIT_RESERVED)) {
            m_hit_fill++;
            m_total_hit_lat += time - mf->get_issue_time(m_core_id<0);
        } else {
            m_fill++;
            m_total_lat += time - mf->get_issue_time(m_core_id<0);
        }
    }
    return idx;
}

unsigned tag_array::fill( mem_fetch *mf, new_addr_type addr, unsigned index, unsigned time ) 
{
    assert( m_config.m_alloc_policy == ON_MISS );
    unsigned idx;
    //if ( index != (unsigned)-1 ) { // lld: -1 means bypass
        cache_request_status status = fill_probe(mf,addr,idx);
        if(status == HIT) // lld: previous request has filled this line
            index = (unsigned)-1;
        else if(status == HIT_RESERVED) {
            if( m_config.m_replacement_policy == SECTOR || m_config.m_replacement_policy == ADAP_GRAN ) {
                for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++) // lld: sector and adaptive grain cache
                    if(mf->get_sector_mask(m_core_id<0).test(i))
                        m_lines[idx].m_ready_sector_mask.set(i);
                //if(m_lines[idx].m_ready_sector_mask == m_lines[idx].m_sector_mask && m_lines[idx].m_status == RESERVED) // lld: sector and adaptive grain cache
                if(m_lines[idx].m_status == RESERVED) // lld: sector and adaptive grain cache
                    m_lines[idx].fill(time, mf->get_wid());
            } else
                m_lines[idx].fill(time, mf->get_wid());
        }
        else if(status == MISS_PARTIAL) { // lld: sector cache
            index = idx;
            for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++) // lld: sector and adaptive grain cache
                if(mf->get_sector_mask(m_core_id<0).test(i)) {
                    //assert(m_lines[idx].m_sector_mask.test(i));
                    //m_lines[idx].m_sector_mask.set(i);
                    //assert(!m_lines[idx].m_ready_sector_mask.test(i));
                    //m_lines[idx].m_ready_sector_mask.set(i);
                    if(m_lines[idx].m_sector_mask.test(i))
                        m_lines[idx].m_ready_sector_mask.set(i);
                }
            //if(m_lines[idx].m_ready_sector_mask == m_lines[idx].m_sector_mask && m_lines[idx].m_status == RESERVED) // lld: sector and adaptive grain cache
            if(m_lines[idx].m_status == RESERVED) // lld: sector and adaptive grain cache
                m_lines[idx].fill(time, mf->get_wid());
        }
        else {
            //assert((status == MISS || status == RESERVATION_FAIL) &&
            //       (mf->get_status(m_core_id<0) == MISS_PARTIAL || mf->get_status(m_core_id<0) == MISS));
            index = (unsigned)-1;
        }
    /*
    }
    else {
        if(probe(mf,addr,idx) == HIT_RESERVED) { // lld: this line is allocated in cache later
            index = idx;
            m_lines[index].fill(time, mf->get_wid());
        }
    }
       */

    // lld: for memory latency calculation
    if( m_type_id != 1 && mf->get_issue(m_core_id<0) )
    {
        mf->set_ret_time(m_core_id<0, time);
        assert(mf->get_issue_time(m_core_id<0));
        if(m_core_id >= 0 && (mf->get_status(true) == HIT || mf->get_status(true) == HIT_RESERVED)) {
            m_hit_fill++;
            m_total_hit_lat += time - mf->get_issue_time(m_core_id<0);
        } else {
            m_fill++;
            m_total_lat += time - mf->get_issue_time(m_core_id<0);
        }
    }
    return index;
}

void tag_array::flush() 
{
    for (unsigned i=0; i < m_config.get_num_lines(); i++) {
        if(m_config.m_replacement_policy == ADAP_GRAN) { // lld
            if(i % m_config.m_assoc < m_config.m_true_assoc)
                m_lines[i].m_status = INVALID;
            else
                m_lines[i].m_status = UNUSED;
        } else
            m_lines[i].m_status = INVALID;
    }
    if(m_config.m_replacement_policy == ADAP_GRAN) { // lld
        for (unsigned i=0; i < m_config.m_nset; i++)
            m_sets[i].reset(m_config.m_assoc);
    }
}

float tag_array::windowed_miss_rate( ) const
{
    unsigned n_access    = m_access - m_prev_snapshot_access;
    unsigned n_miss      = m_miss - m_prev_snapshot_miss;
    // unsigned n_pending_hit = m_pending_hit - m_prev_snapshot_pending_hit;

    float missrate = 0.0f;
    if (n_access != 0)
        missrate = (float) n_miss / n_access;
    return missrate;
}

void tag_array::new_window()
{
    m_prev_snapshot_access = m_access;
    m_prev_snapshot_miss = m_miss;
    m_prev_snapshot_pending_hit = m_pending_hit;
}

void tag_array::print( FILE *stream, unsigned &total_access, unsigned &total_misses ) const
{
    m_config.print(stream);
    fprintf( stream, "\t\tAccess = %d, Miss = %d (%.3g), PendingHit = %d (%.3g)\n", 
             m_access, m_miss, (float) m_miss / m_access, 
             m_pending_hit, (float) m_pending_hit / m_access);
    total_misses+=m_miss;
    total_access+=m_access;
}

void tag_array::get_stats(unsigned &total_access, unsigned &total_misses, unsigned &total_hit_res, unsigned &total_res_fail) const{
    // Update statistics from the tag array
    total_access    = m_access;
    total_misses    = m_miss;
    total_hit_res   = m_pending_hit;
    total_res_fail  = m_res_fail;
}


bool was_write_sent( const std::list<cache_event> &events )
{
    for( std::list<cache_event>::const_iterator e=events.begin(); e!=events.end(); e++ ) {
        if( *e == WRITE_REQUEST_SENT ) 
            return true;
    }
    return false;
}

bool was_writeback_sent( const std::list<cache_event> &events )
{
    for( std::list<cache_event>::const_iterator e=events.begin(); e!=events.end(); e++ ) {
        if( *e == WRITE_BACK_REQUEST_SENT ) 
            return true;
    }
    return false;
}

bool was_read_sent( const std::list<cache_event> &events )
{
    for( std::list<cache_event>::const_iterator e=events.begin(); e!=events.end(); e++ ) {
        if( *e == READ_REQUEST_SENT ) 
            return true;
    }
    return false;
}
/****************************************************************** MSHR ******************************************************************/

/// Checks if there is a pending request to the lower memory level already
// lld: sector cache
bool mshr_table::probe( new_addr_type block_addr, mem_fetch *mf ) const{
    if(m_type == ASSOC) {
        table::const_iterator a = m_data.find(block_addr);
        return a != m_data.end();
    } else if(m_type == SECTOR_MSHR) {
        table::const_iterator a = m_data.find(block_addr);
        if( a != m_data.end() ) {
            for(unsigned i = 0; i < MAX_CACHE_CHUNK_NUM; i++)
                if(mf->get_sector_mask(m_l2).test(i) &&
                   (!a->second.m_sector_mask.test(i) && !a->second.m_original_ready_sector_mask.test(i)))
                    return false;
            return true;
        }
        return false;
    } else if(m_type == LARGE_MSHR) {
        new_addr_type mshr_block_addr = get_block_addr(block_addr);
        table::const_iterator a = m_data.find(mshr_block_addr);
        if( a != m_data.end() ) {
            for(unsigned i = 0; i < m_cache_line_sz/CACHE_CHUNK_SIZE; i++)
                if(!a->second.m_sector_mask.test((block_addr-mshr_block_addr)/CACHE_CHUNK_SIZE+i))
                    return false;
            return true;
        }
        return false;
    } else
        abort();
}

// lld: check if this request has been returned
bool mshr_table::is_ready( new_addr_type block_addr, mem_fetch *mf ) const{
    if(m_type == ASSOC) {
        assert(m_data.find(block_addr) != m_data.end());
        return find(m_current_response.begin(), m_current_response.end(), block_addr) != m_current_response.end();
    } else if(m_type == SECTOR_MSHR) {
        table::const_iterator a = m_data.find(block_addr);
        if( a != m_data.end() ) {
            for(unsigned i = 0; i < MAX_CACHE_CHUNK_NUM; i++)
                //if(mf->get_original_sector_mask(m_l2).test(i) && !a->second.m_ready_sector_mask.test(i))
                if(mf->get_sector_mask(m_l2).test(i) && !a->second.m_ready_sector_mask.test(i))
                    return false;
            return true;
        }
        return false;
    } else if(m_type == LARGE_MSHR) {
        new_addr_type mshr_block_addr = get_block_addr(block_addr);
        table::const_iterator a = m_data.find(mshr_block_addr);
        if( a != m_data.end() ) {
            for(unsigned i = 0; i < m_cache_line_sz/CACHE_CHUNK_SIZE; i++)
                if(!a->second.m_ready_sector_mask.test((block_addr-mshr_block_addr)/CACHE_CHUNK_SIZE+i))
                    return false;
            return true;
        }
        return false;
    } else
        abort();
}

/// Checks if there is space for tracking a new memory access
bool mshr_table::full( new_addr_type block_addr ) {
    if(m_type == LARGE_MSHR)
        block_addr = get_block_addr(block_addr);
    table::const_iterator i=m_data.find(block_addr);
    if ( i != m_data.end() ) {
        if(i->second.m_list.size() >= m_max_merged) m_list_full++;
        return i->second.m_list.size() >= m_max_merged;
    } else {
        if(m_data.size() >= m_num_entries) m_size_full++;
        return m_data.size() >= m_num_entries;
    }
}

/// Add or merge this access
void mshr_table::add( new_addr_type block_addr, mem_fetch *mf, std::bitset<MAX_CACHE_CHUNK_NUM> ready_sector_mask ){
    if(m_type == ASSOC || m_type == SECTOR_MSHR) {
	    m_data[block_addr].m_list.push_back(mf);
	    assert( m_data.size() <= m_num_entries );
	    assert( m_data[block_addr].m_list.size() <= m_max_merged );
	    // indicate that this MSHR entry contains an atomic operation
	    if ( mf->isatomic() ) {
	    	m_data[block_addr].m_has_atomic = true;
	    }
        // lld: sector cache
        if(m_type == SECTOR_MSHR) {
            if(m_data[block_addr].m_list.size() == 1) { // it means this is the first miss for this line
                m_data[block_addr].m_sector_mask.reset();
                m_data[block_addr].m_original_ready_sector_mask = ready_sector_mask;
                m_data[block_addr].m_ready_sector_mask = ready_sector_mask;
            }
            // update sector mask of mf
            for(unsigned i = 0; i < MAX_CACHE_CHUNK_NUM; i++)
                //if(m_data[block_addr].m_sector_mask.test(i))
                if(m_data[block_addr].m_sector_mask.test(i) || m_data[block_addr].m_ready_sector_mask.test(i))
                    mf->set_sector_mask(m_l2, i, false);
            for(unsigned i = 0; i < MAX_CACHE_CHUNK_NUM; i++)
                if(mf->get_sector_mask(m_l2).test(i))
                    m_data[block_addr].m_sector_mask.set(i);
        }
    } else if(m_type == LARGE_MSHR) {
        new_addr_type mshr_block_addr = get_block_addr(block_addr);
	    m_data[mshr_block_addr].m_list.push_back(mf);
	    assert( m_data.size() <= m_num_entries );
	    assert( m_data[mshr_block_addr].m_list.size() <= m_max_merged );
	    // indicate that this MSHR entry contains an atomic operation
	    if ( mf->isatomic() ) {
	    	m_data[mshr_block_addr].m_has_atomic = true;
	    }

        if(m_data[mshr_block_addr].m_list.size() == 1) { // it means this is the first miss for this line
            m_data[mshr_block_addr].m_sector_mask.reset();
            m_data[mshr_block_addr].m_ready_sector_mask.reset();
        }
        for(unsigned i = 0; i < m_line_sz/CACHE_CHUNK_SIZE; i++) {
            if(i < m_cache_line_sz/CACHE_CHUNK_SIZE)
                mf->set_sector_mask(m_l2, i, true);
            else
                mf->set_sector_mask(m_l2, i, false);
        }
        for(unsigned i = 0; i < m_cache_line_sz/CACHE_CHUNK_SIZE; i++)
            m_data[mshr_block_addr].m_sector_mask.set((block_addr-mshr_block_addr)/CACHE_CHUNK_SIZE+i);
    } else
        abort();
}

/// Accept a new cache fill response: mark entry ready for processing
// lld: sector cache
void mshr_table::mark_ready( mem_fetch *mf, new_addr_type block_addr, bool &has_atomic ){
    assert( !busy() );
    if(m_type == ASSOC) {
        table::iterator a = m_data.find(block_addr);
        assert( a != m_data.end() ); // don't remove same request twice
        m_current_response.push_back( block_addr );
        has_atomic = a->second.m_has_atomic;
        assert( m_current_response.size() <= m_data.size() );
    } else if(m_type == SECTOR_MSHR) {
        table::iterator a = m_data.find(block_addr);
        assert( a != m_data.end() ); // don't remove same request twice
        has_atomic = a->second.m_has_atomic;
        mshr_entry *cur_entry = &a->second;

        // mark sector ready
        for(unsigned i = 0; i < MAX_CACHE_CHUNK_NUM; i++)
            if(mf->get_sector_mask(m_l2).test(i))
                cur_entry->m_ready_sector_mask.set(i);

        // find ready requests
        assert( !cur_entry->m_list.empty() );
        mem_fetch *result;
        list<mem_fetch*>::iterator it;
        for(it = cur_entry->m_list.begin(); it != cur_entry->m_list.end();)
        {
            result = *it;
            bool found = true;
            for(unsigned i = 0; i < MAX_CACHE_CHUNK_NUM; i++)
                if(result->get_sector_mask(m_l2).test(i) && !cur_entry->m_ready_sector_mask.test(i)) {
                    found = false;
                    break;
                }
            if(found) {
                assert(!result->get_issue(m_l2) || result->get_ret_time(m_l2) || mf == result);
                if(!result->get_prefetch(m_l2))
                    m_current_ready_request.push_back(result);
                else
                    m_other_ready_request.push_back(result);
                it++;
                list<mem_fetch*>::iterator next_it = it;
                it--;
                cur_entry->m_list.erase(it);
                it = next_it;
            }
            else
                it++;
        }
        if ( cur_entry->m_list.empty() ) {
            // release entry
            for(unsigned i = 0; i < MAX_CACHE_CHUNK_NUM; i++)
                assert((cur_entry->m_ready_sector_mask.test(i) &&
                        (cur_entry->m_sector_mask.test(i) || cur_entry->m_original_ready_sector_mask.test(i)))
                       || (!cur_entry->m_ready_sector_mask.test(i) &&
                           !cur_entry->m_sector_mask.test(i) && !cur_entry->m_original_ready_sector_mask.test(i)));
            m_data.erase(block_addr);
        }
    } else if(m_type == LARGE_MSHR) {
        new_addr_type new_block_addr = get_cache_block_addr(mf->get_addr());
        new_addr_type mshr_block_addr = get_block_addr(new_block_addr);
        table::iterator a = m_data.find(mshr_block_addr);
        assert( a != m_data.end() ); // don't remove same request twice
        has_atomic = a->second.m_has_atomic;
        mshr_entry *cur_entry = &a->second;

        // mark sector ready
        for(unsigned i = 0; i < m_line_sz/CACHE_CHUNK_SIZE; i++) {
            if((new_block_addr-mshr_block_addr)/CACHE_CHUNK_SIZE+i >= m_line_sz/CACHE_CHUNK_SIZE)
                break;
            if(mf->get_sector_mask(m_l2).test(i)) {
                assert(!cur_entry->m_ready_sector_mask.test((new_block_addr-mshr_block_addr)/CACHE_CHUNK_SIZE+i));
                cur_entry->m_ready_sector_mask.set((new_block_addr-mshr_block_addr)/CACHE_CHUNK_SIZE+i);
            }
        }

        // find ready requests
        assert( !cur_entry->m_list.empty() );
        mem_fetch *result;
        list<mem_fetch*>::iterator it;
        bool found_one = false;
        for(it = cur_entry->m_list.begin(); it != cur_entry->m_list.end();)
        {
            result = *it;
            bool found = true;
            new_addr_type cur_block_addr = get_cache_block_addr(result->get_addr());
            for(unsigned i = 0; i < m_line_sz/CACHE_CHUNK_SIZE; i++) {
                if((cur_block_addr-mshr_block_addr)/CACHE_CHUNK_SIZE+i >= m_line_sz/CACHE_CHUNK_SIZE)
                    break;
                if(result->get_sector_mask(m_l2).test(i) && !cur_entry->m_ready_sector_mask.test((cur_block_addr-mshr_block_addr)/CACHE_CHUNK_SIZE+i)) {
                    found = false;
                    break;
                }
            }
            if(found) {
                found_one = true;
                assert(!result->get_issue(m_l2) || mf == result);
                if(!result->get_prefetch(m_l2))
                    m_current_ready_request.push_back(result);
                else
                    m_other_ready_request.push_back(result);
                it++;
                list<mem_fetch*>::iterator next_it = it;
                it--;
                cur_entry->m_list.erase(it);
                it = next_it;
            }
            else
                it++;
        }
        assert(found_one);
        if ( cur_entry->m_list.empty() ) {
            // release entry
            for(unsigned i = 0; i < MAX_CACHE_CHUNK_NUM; i++)
                assert((cur_entry->m_ready_sector_mask.test(i) && cur_entry->m_sector_mask.test(i))
                       || (!cur_entry->m_ready_sector_mask.test(i) && !cur_entry->m_sector_mask.test(i)));
            m_data.erase(mshr_block_addr);
        }
    } else
        abort();
}

/// Returns next ready access
mem_fetch *mshr_table::next_access(){
    // lld
    if(m_type == SECTOR_MSHR || m_type == LARGE_MSHR) {
        assert( access_ready() );
        // delete prefetch requests
        for(list<mem_fetch*>::iterator it = m_other_ready_request.begin(); it != m_other_ready_request.end(); it++)
            delete *it;
        m_other_ready_request.clear();
        mem_fetch *result = m_current_ready_request.front();
        m_current_ready_request.pop_front();
        return result;
    } else { // FIXME: normal mshr cannot deal with prefetch
        assert( access_ready() );
        new_addr_type block_addr = m_current_response.front();
        assert( !m_data[block_addr].m_list.empty() );
        mem_fetch *result = m_data[block_addr].m_list.front();
        m_data[block_addr].m_list.pop_front();
        if ( m_data[block_addr].m_list.empty() ) {
            // release entry
            m_data.erase(block_addr);
            m_current_response.pop_front();
        }
        return result;
    }
}

// lld: MSHR stats
void mshr_table::get_sub_stats(struct cache_sub_stats &css) const {
    struct cache_sub_stats t_css;
    t_css.clear();
    t_css.mshr_size_full = m_size_full;
    t_css.mshr_list_full = m_list_full;
    css += t_css;
}

void mshr_table::display( FILE *fp ) const{
    fprintf(fp,"MSHR contents\n");
    for ( table::const_iterator e=m_data.begin(); e!=m_data.end(); ++e ) {
        unsigned block_addr = e->first;
        fprintf(fp,"MSHR: tag=0x%06x, atomic=%d %zu entries : ", block_addr, e->second.m_has_atomic, e->second.m_list.size());
        if ( !e->second.m_list.empty() ) {
            mem_fetch *mf = e->second.m_list.front();
            fprintf(fp,"%p :",mf);
            mf->print(fp);
        } else {
            fprintf(fp," no memory requests???\n");
        }
    }
}
/***************************************************************** Caches *****************************************************************/
cache_stats::cache_stats(){
    m_stats.resize(NUM_MEM_ACCESS_TYPE);
    for(unsigned i=0; i<NUM_MEM_ACCESS_TYPE; ++i){
        m_stats[i].resize(NUM_CACHE_REQUEST_STATUS, 0);
    }
    m_cache_port_available_cycles = 0; 
    m_cache_data_port_busy_cycles = 0; 
    m_cache_fill_port_busy_cycles = 0; 
}

void cache_stats::clear(){
    ///
    /// Zero out all current cache statistics
    ///
    for(unsigned i=0; i<NUM_MEM_ACCESS_TYPE; ++i){
        std::fill(m_stats[i].begin(), m_stats[i].end(), 0);
    }
    m_cache_port_available_cycles = 0; 
    m_cache_data_port_busy_cycles = 0; 
    m_cache_fill_port_busy_cycles = 0; 
}

void cache_stats::inc_stats(int access_type, int access_outcome){
    ///
    /// Increment the stat corresponding to (access_type, access_outcome) by 1.
    ///
    if(!check_valid(access_type, access_outcome))
        assert(0 && "Unknown cache access type or access outcome");

    m_stats[access_type][access_outcome]++;
}

enum cache_request_status cache_stats::select_stats_status(enum cache_request_status probe, enum cache_request_status access) const {
	///
	/// This function selects how the cache access outcome should be counted. HIT_RESERVED is considered as a MISS
	/// in the cores, however, it should be counted as a HIT_RESERVED in the caches.
	///
	if(probe == HIT_RESERVED && access != RESERVATION_FAIL)
		return probe;
    else if(probe == MISS_PARTIAL && access != RESERVATION_FAIL) // lld: sector cache
		return probe;
	else
		return access;
}

unsigned &cache_stats::operator()(int access_type, int access_outcome){
    ///
    /// Simple method to read/modify the stat corresponding to (access_type, access_outcome)
    /// Used overloaded () to avoid the need for separate read/write member functions
    ///
    if(!check_valid(access_type, access_outcome))
        assert(0 && "Unknown cache access type or access outcome");

    return m_stats[access_type][access_outcome];
}

unsigned cache_stats::operator()(int access_type, int access_outcome) const{
    ///
    /// Const accessor into m_stats.
    ///
    if(!check_valid(access_type, access_outcome))
        assert(0 && "Unknown cache access type or access outcome");

    return m_stats[access_type][access_outcome];
}

cache_stats cache_stats::operator+(const cache_stats &cs){
    ///
    /// Overloaded + operator to allow for simple stat accumulation
    ///
    cache_stats ret;
    for(unsigned type=0; type<NUM_MEM_ACCESS_TYPE; ++type){
        for(unsigned status=0; status<NUM_CACHE_REQUEST_STATUS; ++status){
            ret(type, status) = m_stats[type][status] + cs(type, status);
        }
    }
    ret.m_cache_port_available_cycles = m_cache_port_available_cycles + cs.m_cache_port_available_cycles; 
    ret.m_cache_data_port_busy_cycles = m_cache_data_port_busy_cycles + cs.m_cache_data_port_busy_cycles; 
    ret.m_cache_fill_port_busy_cycles = m_cache_fill_port_busy_cycles + cs.m_cache_fill_port_busy_cycles; 
    return ret;
}

cache_stats &cache_stats::operator+=(const cache_stats &cs){
    ///
    /// Overloaded += operator to allow for simple stat accumulation
    ///
    for(unsigned type=0; type<NUM_MEM_ACCESS_TYPE; ++type){
        for(unsigned status=0; status<NUM_CACHE_REQUEST_STATUS; ++status){
            m_stats[type][status] += cs(type, status);
        }
    }
    m_cache_port_available_cycles += cs.m_cache_port_available_cycles; 
    m_cache_data_port_busy_cycles += cs.m_cache_data_port_busy_cycles; 
    m_cache_fill_port_busy_cycles += cs.m_cache_fill_port_busy_cycles; 
    return *this;
}

void cache_stats::print_stats(FILE *fout, const char *cache_name) const{
    ///
    /// Print out each non-zero cache statistic for every memory access type and status
    /// "cache_name" defaults to "Cache_stats" when no argument is provided, otherwise
    /// the provided name is used.
    /// The printed format is "<cache_name>[<request_type>][<request_status>] = <stat_value>"
    ///
    std::string m_cache_name = cache_name;
    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
            if(m_stats[type][status] > 0){
                fprintf(fout, "\t%s[%s][%s] = %u\n",
                    m_cache_name.c_str(),
                    mem_access_type_str((enum mem_access_type)type),
                    cache_request_status_str((enum cache_request_status)status),
                    m_stats[type][status]);
            }
        }
    }
}

void cache_sub_stats::print_port_stats(FILE *fout, const char *cache_name) const
{
    float data_port_util = 0.0f; 
    if (port_available_cycles > 0) {
        data_port_util = (float) data_port_busy_cycles / port_available_cycles; 
    }
    fprintf(fout, "%s_data_port_util = %.3f\n", cache_name, data_port_util); 
    float fill_port_util = 0.0f; 
    if (port_available_cycles > 0) {
        fill_port_util = (float) fill_port_busy_cycles / port_available_cycles; 
    }
    fprintf(fout, "%s_fill_port_util = %.3f\n", cache_name, fill_port_util); 
}

unsigned cache_stats::get_stats(enum mem_access_type *access_type, unsigned num_access_type, enum cache_request_status *access_status, unsigned num_access_status) const{
    ///
    /// Returns a sum of the stats corresponding to each "access_type" and "access_status" pair.
    /// "access_type" is an array of "num_access_type" mem_access_types.
    /// "access_status" is an array of "num_access_status" cache_request_statuses.
    ///
    unsigned total=0;
    for(unsigned type =0; type < num_access_type; ++type){
        for(unsigned status=0; status < num_access_status; ++status){
            if(!check_valid((int)access_type[type], (int)access_status[status]))
                assert(0 && "Unknown cache access type or access outcome");
            total += m_stats[access_type[type]][access_status[status]];
        }
    }
    return total;
}
void cache_stats::get_sub_stats(struct cache_sub_stats &css) const{
    ///
    /// Overwrites "css" with the appropriate statistics from this cache.
    ///
    struct cache_sub_stats t_css;
    t_css.clear();

    for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
        if(type == L1_PREF_ACC_R || type == L2_PREF_ACC_R ||
           type == L1_WRBK_ACC || type == L2_WRBK_ACC ||
           type == L1_WR_ALLOC_R || type == L2_WR_ALLOC_R ||
           type == GLOBAL_ACC_W || type == LOCAL_ACC_W) // lld: these accesses should not affect performance
            continue;
        for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS; ++status) {
            if(status == HIT || status == MISS || status == HIT_RESERVED)
                t_css.accesses += m_stats[type][status];

            if(status == MISS)
                t_css.misses += m_stats[type][status];

            if(status == MISS_PARTIAL) { // lld: sector cache
                t_css.accesses += m_stats[type][status];
                t_css.partial_misses += m_stats[type][status];
            }

            if(status == HIT_RESERVED)
                t_css.pending_hits += m_stats[type][status];

            if(status == RESERVATION_FAIL)
                t_css.res_fails += m_stats[type][status];
        }
    }

    t_css.port_available_cycles = m_cache_port_available_cycles; 
    t_css.data_port_busy_cycles = m_cache_data_port_busy_cycles; 
    t_css.fill_port_busy_cycles = m_cache_fill_port_busy_cycles; 

    css = t_css;
}

bool cache_stats::check_valid(int type, int status) const{
    ///
    /// Verify a valid access_type/access_status
    ///
    if((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (status >= 0) && (status < NUM_CACHE_REQUEST_STATUS))
        return true;
    else
        return false;
}

void cache_stats::sample_cache_port_utility(bool data_port_busy, bool fill_port_busy) 
{
    m_cache_port_available_cycles += 1; 
    if (data_port_busy) {
        m_cache_data_port_busy_cycles += 1; 
    } 
    if (fill_port_busy) {
        m_cache_fill_port_busy_cycles += 1; 
    } 
}

baseline_cache::bandwidth_management::bandwidth_management(cache_config &config) 
: m_config(config)
{
    m_data_port_occupied_cycles = 0; 
    m_fill_port_occupied_cycles = 0; 
}

/// use the data port based on the outcome and events generated by the mem_fetch request 
void baseline_cache::bandwidth_management::use_data_port(mem_fetch *mf, enum cache_request_status outcome, const std::list<cache_event> &events)
{
    unsigned data_size = mf->get_data_size(); 
    unsigned port_width = m_config.m_data_port_width; 
    switch (outcome) {
    case HIT: {
        unsigned data_cycles = data_size / port_width + ((data_size % port_width > 0)? 1 : 0); 
        m_data_port_occupied_cycles += data_cycles; 
        } break; 
    case HIT_RESERVED: 
    case MISS_PARTIAL: // lld: sector cache
    case MISS: {
        // the data array is accessed to read out the entire line for write-back 
        if (was_writeback_sent(events)) {
            unsigned data_cycles = m_config.m_line_sz / port_width; 
            m_data_port_occupied_cycles += data_cycles; 
        }
        } break; 
    case RESERVATION_FAIL: 
        // Does not consume any port bandwidth 
        break; 
    default: 
        assert(0); 
        break; 
    } 
}

/// use the fill port 
void baseline_cache::bandwidth_management::use_fill_port(mem_fetch *mf)
{
    // assume filling the entire line with the returned request 
    // FIXME: lld: may not fill the whole line
    unsigned fill_cycles = m_config.m_line_sz / m_config.m_data_port_width; 
    m_fill_port_occupied_cycles += fill_cycles; 
}

/// called every cache cycle to free up the ports 
void baseline_cache::bandwidth_management::replenish_port_bandwidth()
{
    if (m_data_port_occupied_cycles > 0) {
        m_data_port_occupied_cycles -= 1; 
    }
    assert(m_data_port_occupied_cycles >= 0); 

    if (m_fill_port_occupied_cycles > 0) {
        m_fill_port_occupied_cycles -= 1; 
    }
    assert(m_fill_port_occupied_cycles >= 0); 
}

/// query for data port availability 
bool baseline_cache::bandwidth_management::data_port_free() const
{
    return (m_data_port_occupied_cycles == 0); 
}

/// query for fill port availability 
bool baseline_cache::bandwidth_management::fill_port_free() const
{
    return (m_fill_port_occupied_cycles == 0); 
}

/// Sends next request to lower level of memory
void baseline_cache::cycle(){
    if ( !m_miss_queue.empty() ) {
        mem_fetch *mf = m_miss_queue.front();
        if ( !m_memport->full(mf->size(),mf->get_is_write()) ) {
            m_miss_queue.pop_front();
            m_memport->push(mf);
        }
    }
    bool data_port_busy = !m_bandwidth_management.data_port_free(); 
    bool fill_port_busy = !m_bandwidth_management.fill_port_free(); 
    m_stats.sample_cache_port_utility(data_port_busy, fill_port_busy); 
    m_bandwidth_management.replenish_port_bandwidth(); 
}

/// Interface for response from lower memory level (model bandwidth restictions in caller)
void baseline_cache::fill(mem_fetch *mf, unsigned time){
    extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
    assert( e != m_extra_mf_fields.end() );
    assert( e->second.m_valid );
    mf->set_data_size( e->second.m_data_size );
    unsigned idx; // lld: record whether miss is bypassed
    if ( m_config.m_alloc_policy == ON_MISS )
        idx = m_tag_array->fill(mf,e->second.m_block_addr,e->second.m_cache_index,time);
    else if ( m_config.m_alloc_policy == ON_FILL )
        idx = m_tag_array->fill(mf,e->second.m_block_addr,time);
    else abort();
    bool has_atomic = false;
    m_mshrs.mark_ready(mf, e->second.m_block_addr, has_atomic);
    if (has_atomic) {
        assert(m_config.m_alloc_policy == ON_MISS);
        if(idx != (unsigned)-1) { // lld
            cache_block_t &block = m_tag_array->get_block(e->second.m_cache_index);
            block.m_status = MODIFIED; // mark line as dirty for atomic operation
        }
    }
    m_extra_mf_fields.erase(mf);
    if(mf->get_prefetch(m_l2)) { // lld: prefetch requests
        assert(find(m_extra_pending_list.begin(), m_extra_pending_list.end(), mf) != m_extra_pending_list.end());
        m_extra_pending_list.remove(mf);
    }

    //if(g_debug_execution >= 1) {
    //    cerr << dec << m_name << " fills " << idx << ", 0x" << hex << e->second.m_block_addr << endl;
    //    //if(e->second.m_block_addr == 0x8004d380)
    //    //    mf->print(stderr, true);
    //}
    // lld: the fill port is not used on bypass
    if(idx != (unsigned)-1)
        m_bandwidth_management.use_fill_port(mf);

    // fill for each merged request FIXME
    if(!m_l2) {
        for(list<mem_fetch*>::iterator it = mf->m_merged_requests.begin(); it != mf->m_merged_requests.end(); it++) {
            mem_fetch *merged_mf = *it;
            extra_mf_fields_lookup::iterator ee = m_extra_mf_fields.find(merged_mf);
            assert( ee != m_extra_mf_fields.end() );
            assert( ee->second.m_valid );
            merged_mf->set_data_size( ee->second.m_data_size );
            unsigned idx; // lld: record whether miss is bypassed
            if ( m_config.m_alloc_policy == ON_MISS )
                idx = m_tag_array->fill(merged_mf,ee->second.m_block_addr,ee->second.m_cache_index,time);
            else if ( m_config.m_alloc_policy == ON_FILL )
                idx = m_tag_array->fill(merged_mf,ee->second.m_block_addr,time);
            else abort();
            if (has_atomic) {
                assert(m_config.m_alloc_policy == ON_MISS);
                if(idx != (unsigned)-1) { // lld
                    cache_block_t &block = m_tag_array->get_block(ee->second.m_cache_index);
                    block.m_status = MODIFIED; // mark line as dirty for atomic operation
                }
            }
            m_extra_mf_fields.erase(merged_mf);
            if(merged_mf->get_prefetch(m_l2)) { // lld: prefetch requests
                assert(find(m_extra_pending_list.begin(), m_extra_pending_list.end(), merged_mf) != m_extra_pending_list.end());
                m_extra_pending_list.remove(merged_mf);
            }

            // lld: the fill port is not used on bypass
            if(idx != (unsigned)-1)
                m_bandwidth_management.use_fill_port(merged_mf);
        }
    }
}

/// Checks if mf is waiting to be filled by lower memory level
bool baseline_cache::waiting_for_fill( mem_fetch *mf ){
    extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
    return e != m_extra_mf_fields.end();
}

void baseline_cache::print(FILE *fp, unsigned &accesses, unsigned &misses) const{
    fprintf( fp, "Cache %s:\t", m_name.c_str() );
    m_tag_array->print(fp,accesses,misses);
}

// lld: print replacement stats
void baseline_cache::print_stats() const{
    if( m_config.m_custom_repl == 1 )
        m_tag_array->print_stats(cout, m_name);
}

void baseline_cache::display_state( FILE *fp ) const{
    fprintf(fp,"Cache %s:\n", m_name.c_str() );
    m_mshrs.display(fp);
    fprintf(fp,"\n");
}

/// Read miss handler without writeback
void baseline_cache::send_read_request(new_addr_type addr, new_addr_type block_addr, unsigned cache_index, mem_fetch *mf,
		unsigned time, bool &do_miss, std::list<cache_event> &events, bool read_only, bool wa){

	bool wb=false;
	cache_block_t e;
	send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb, e, events, read_only, wa);
}

/// Read miss handler. Check MSHR hit or MSHR available
void baseline_cache::send_read_request(new_addr_type addr, new_addr_type block_addr, unsigned cache_index, mem_fetch *mf,
		unsigned time, bool &do_miss, bool &wb, cache_block_t &evicted, std::list<cache_event> &events, bool read_only, bool wa){

    bool mshr_hit;
    mshr_hit = m_mshrs.probe(block_addr,mf);
    bool mshr_avail = !m_mshrs.full(block_addr);
    if ( mshr_hit && mshr_avail ) {
        if(m_mshrs.is_ready(block_addr,mf)) { // lld: prevent pending returned requests from allocating cache space
            do_miss = false;
            wb = true;
            return;
        }
        if(mf->get_prefetch(m_l2)) {
            do_miss = false;
            wb = true;
            return;
        }

    	if(read_only)
    		m_tag_array->access(mf,block_addr,time,cache_index);
    	else
    		m_tag_array->access(mf,block_addr,time,cache_index,wb,evicted);

        if(mf->get_prefetch(m_l2) && cache_index == (unsigned)-1) { // lld: discard bypassed prefetch
            do_miss = false;
            wb = true;
        } else if(!mf->get_done(m_l2)){ // additional replacement is needed
            do_miss = true;
            return;
        } else {
            bitset<MAX_CACHE_CHUNK_NUM> sector_mask;
            if((m_config.m_replacement_policy == SECTOR || m_config.m_replacement_policy == ADAP_GRAN) && cache_index != (unsigned)-1)
                sector_mask = m_tag_array->get_line(cache_index)->m_ready_sector_mask; // lld: sector and adaptive grain cache
            m_mshrs.add(block_addr,mf,sector_mask);
            do_miss = true;
        }
    } else if ( !mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size) ) {
    	if(read_only)
    		m_tag_array->access(mf,block_addr,time,cache_index);
    	else
    		m_tag_array->access(mf,block_addr,time,cache_index,wb,evicted);

        if(mf->get_prefetch(m_l2) && cache_index == (unsigned)-1) { // lld: discard bypassed prefetch
            do_miss = false;
            wb = true;
        } else if(!mf->get_done(m_l2)){ // additional replacement is needed
            do_miss = true;
            return;
        } else {
            bitset<MAX_CACHE_CHUNK_NUM> sector_mask;
            if((m_config.m_replacement_policy == SECTOR || m_config.m_replacement_policy == ADAP_GRAN) && cache_index != (unsigned)-1)
                sector_mask = m_tag_array->get_line(cache_index)->m_ready_sector_mask; // lld: sector and adaptive grain cache
            m_mshrs.add(block_addr,mf,sector_mask);
            issue_request(block_addr, cache_index, mf, time, events, wa);
            //m_extra_mf_fields[mf] = extra_mf_fields(block_addr,cache_index, mf->get_data_size());
            //if(m_config.m_mshr_type == SECTOR_MSHR) // lld: sector cache
            //    mf->set_data_size( mf->get_sector_mask(m_l2).count()*CACHE_CHUNK_SIZE );
            //else
            //    mf->set_data_size( m_config.get_line_sz() );
            //m_miss_queue.push_back(mf);
            //mf->set_issue(m_l2);
            //mf->set_issue_time(m_l2, time); // lld: for memory latency calculation
            //mf->set_status(m_miss_queue_status,time);
            //if(!wa)
            //	events.push_back(READ_REQUEST_SENT);
            do_miss = true;
            //if(m_name == "L1D_005" || m_name == "L2_bank_003")
            //{
            //    cerr << dec << m_name << " issues " << cache_index << ", 0x" << hex << addr << endl;
            //    if(block_addr == 0x8004d380)
            //        mf->print(stderr, true);
            //}
        }
    }
}


/// Sends write request to lower level memory (write or writeback)
void data_cache::send_write_request(mem_fetch *mf, cache_event request, unsigned time, std::list<cache_event> &events){
    events.push_back(request);
    m_miss_queue.push_back(mf);
    if( m_config.m_mshr_type == SECTOR_MSHR ) // lld: sector cache
        mf->set_data_size( mf->get_sector_mask(m_l2).count()*CACHE_CHUNK_SIZE );
    mf->set_issue_time(m_l2, time); // lld: for memory latency calculation
    mf->set_status(m_miss_queue_status,time);
}


/****** Write-hit functions (Set by config file) ******/

/// Write-back hit: Mark block as modified
cache_request_status data_cache::wr_hit_wb(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status ){
	new_addr_type block_addr = m_config.block_addr(addr);
	m_tag_array->access(mf,block_addr,time,cache_index); // update LRU state
	cache_block_t &block = m_tag_array->get_block(cache_index);
	block.m_status = MODIFIED;

	return HIT;
}

/// Write-through hit: Directly send request to lower level memory
cache_request_status data_cache::wr_hit_wt(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status ){
	if(miss_queue_full(0))
		return RESERVATION_FAIL; // cannot handle request this cycle

	new_addr_type block_addr = m_config.block_addr(addr);
	m_tag_array->access(mf,block_addr,time,cache_index); // update LRU state
	cache_block_t &block = m_tag_array->get_block(cache_index);
	block.m_status = MODIFIED;

	// generate a write-through
	send_write_request(mf, WRITE_REQUEST_SENT, time, events);

	return HIT;
}

/// Write-evict hit: Send request to lower level memory and invalidate corresponding block
cache_request_status data_cache::wr_hit_we(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status ){
	if(miss_queue_full(0))
		return RESERVATION_FAIL; // cannot handle request this cycle

	// generate a write-through/evict
	cache_block_t &block = m_tag_array->get_block(cache_index);
	send_write_request(mf, WRITE_REQUEST_SENT, time, events);

	// Invalidate block
	block.m_status = INVALID;

	return HIT;
}

/// Global write-evict, local write-back: Useful for private caches
enum cache_request_status data_cache::wr_hit_global_we_local_wb(new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, enum cache_request_status status ){
	bool evict = (mf->get_access_type() == GLOBAL_ACC_W); // evict a line that hits on global memory write
	if(evict)
		return wr_hit_we(addr, cache_index, mf, time, events, status); // Write-evict
	else
		return wr_hit_wb(addr, cache_index, mf, time, events, status); // Write-back
}

/****** Write-miss functions (Set by config file) ******/

/// Write-allocate miss: Send write request to lower level memory
// and send a read request for the same block
enum cache_request_status
data_cache::wr_miss_wa( new_addr_type addr,
                        unsigned cache_index, mem_fetch *mf,
                        unsigned time, std::list<cache_event> &events,
                        enum cache_request_status status )
{
    new_addr_type block_addr = m_config.block_addr(addr);

    // Write allocate, maximum 3 requests (write miss, read request, write back request)
    // Conservatively ensure the worst-case request can be handled this cycle
    bool mshr_hit;
    mshr_hit = m_mshrs.probe(block_addr,mf);
    bool mshr_avail = !m_mshrs.full(block_addr);
    if(miss_queue_full(2) 
        || (!(mshr_hit && mshr_avail) 
        && !(!mshr_hit && mshr_avail 
        && (m_miss_queue.size() < m_config.m_miss_queue_size))))
        return RESERVATION_FAIL;

    //send_write_request(mf, WRITE_REQUEST_SENT, time, events);
    // Tries to send write allocate request, returns true on success and false on failure
    //if(!send_write_allocate(mf, addr, block_addr, cache_index, time, events))
    //    return RESERVATION_FAIL;
    if(!mf->get_write_sent(m_l2)) { // lld: prevent redundant write requests
        send_write_request(mf, WRITE_REQUEST_SENT, time, events);
        mf->set_write_sent(m_l2);
    }

    const mem_access_t *ma = new  mem_access_t( m_wr_alloc_type,
                        mf->get_addr(),
                        mf->get_data_size(),
                        false, // Now performing a read
                        mf->get_access_warp_mask(),
                        mf->get_access_byte_mask() );

    mem_fetch *n_mf = new mem_fetch( *ma,
                    NULL,
                    mf->get_ctrl_size(),
                    mf->get_wid(),
                    mf->get_sid(),
                    mf->get_tpc(),
                    mf->get_mem_config());

    bool do_miss = false;
    bool wb = false;
    cache_block_t evicted;

    // Send read request resulting from write miss
    send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
        evicted, events, false, true);

    if(!do_miss && wb) // lld: bypassed prefetch or pending returned requests
        return HIT;

    if( do_miss ){
        // If evicted block is modified and not a write-through
        // (already modified lower level)
        if( wb && (m_config.m_write_policy != WRITE_THROUGH) ) { 
            mem_fetch *wb = m_memfetch_creator->alloc(evicted.m_block_addr,
                m_wrbk_type,m_config.get_line_sz(),true);
            if( m_config.m_mshr_type == SECTOR_MSHR ) // lld: sector cache
                wb->set_data_size( evicted.m_ready_sector_mask.count() );
            m_miss_queue.push_back(wb);
            wb->set_issue_time(m_l2, time); // lld: for memory latency calculation
            wb->set_status(m_miss_queue_status,time);
        }
        if(!mf->get_done(m_l2)) // lld: additional replacement is needed
            return RESERVATION_FAIL;
        return MISS;
    }

    return RESERVATION_FAIL;
}

/// No write-allocate miss: Simply send write request to lower level memory
enum cache_request_status
data_cache::wr_miss_no_wa( new_addr_type addr,
                           unsigned cache_index,
                           mem_fetch *mf,
                           unsigned time,
                           std::list<cache_event> &events,
                           enum cache_request_status status )
{
    if(miss_queue_full(0))
        return RESERVATION_FAIL; // cannot handle request this cycle

    // on miss, generate write through (no write buffering -- too many threads for that)
    send_write_request(mf, WRITE_REQUEST_SENT, time, events);

    return MISS;
}

/****** Read hit functions (Set by config file) ******/

/// Baseline read hit: Update LRU status of block.
// Special case for atomic instructions -> Mark block as modified
enum cache_request_status
data_cache::rd_hit_base( new_addr_type addr,
                         unsigned cache_index,
                         mem_fetch *mf,
                         unsigned time,
                         std::list<cache_event> &events,
                         enum cache_request_status status )
{
    new_addr_type block_addr = m_config.block_addr(addr);
    m_tag_array->access(mf,block_addr,time,cache_index);
    // Atomics treated as global read/write requests - Perform read, mark line as
    // MODIFIED
    if(mf->isatomic()){ 
        assert(mf->get_access_type() == GLOBAL_ACC_R);
        cache_block_t &block = m_tag_array->get_block(cache_index);
        block.m_status = MODIFIED;  // mark line as dirty
    }
    return HIT;
}

/****** Read miss functions (Set by config file) ******/

/// Baseline read miss: Send read request to lower level memory,
// perform write-back as necessary
enum cache_request_status
data_cache::rd_miss_base( new_addr_type addr,
                          unsigned cache_index,
                          mem_fetch *mf,
                          unsigned time,
                          std::list<cache_event> &events,
                          enum cache_request_status status ){
    if(miss_queue_full(1))
        // cannot handle request this cycle
        // (might need to generate two requests)
        return RESERVATION_FAIL; 

    new_addr_type block_addr = m_config.block_addr(addr);
    bool do_miss = false;
    bool wb = false;
    cache_block_t evicted;
    send_read_request( addr,
                       block_addr,
                       cache_index,
                       mf, time, do_miss, wb, evicted, events, false, false);

    if(!do_miss && wb) // lld: bypassed prefetch or pending returned requests
        return HIT;

    if( do_miss ){
        // If evicted block is modified and not a write-through
        // (already modified lower level)
        if(wb && (m_config.m_write_policy != WRITE_THROUGH) ){ 
            mem_fetch *wb = m_memfetch_creator->alloc(evicted.m_block_addr,
                m_wrbk_type,m_config.get_line_sz(),true);
        if( m_config.m_mshr_type == SECTOR_MSHR ) // lld: sector cache
            for(unsigned i = 0; i < MAX_CACHE_CHUNK_NUM; i++)
                wb->set_sector_mask(m_l2, i, evicted.m_ready_sector_mask.test(i));
        send_write_request(wb, WRITE_BACK_REQUEST_SENT, time, events);
    }
        if(!mf->get_done(m_l2)) // lld: additional replacement is needed
            return RESERVATION_FAIL;
        return MISS;
    }
    return RESERVATION_FAIL;
}

/// Access cache for read_only_cache: returns RESERVATION_FAIL if
// request could not be accepted (for any reason)
enum cache_request_status
read_only_cache::access( new_addr_type addr,
                         mem_fetch *mf,
                         unsigned time,
                         std::list<cache_event> &events )
{
    assert( mf->get_data_size() <= m_config.get_line_sz());
    assert(m_config.m_write_policy == READ_ONLY);
    assert(!mf->get_is_write());
    new_addr_type block_addr = m_config.block_addr(addr);
    unsigned cache_index = (unsigned)-1;
    enum cache_request_status status = m_tag_array->probe(mf,block_addr,cache_index);
    enum cache_request_status cache_status = RESERVATION_FAIL;

    if ( status == HIT ) {
        cache_status = m_tag_array->access(mf,block_addr,time,cache_index); // update LRU state
    }else if ( status != RESERVATION_FAIL ) {
        if(!miss_queue_full(0)){
            bool do_miss=false;
            send_read_request(addr, block_addr, cache_index, mf, time, do_miss, events, true, false); // lld: do not prefetch for read_only_cache
            if(do_miss)
                cache_status = MISS;
            else
                cache_status = RESERVATION_FAIL;
        }else{
            cache_status = RESERVATION_FAIL;
        }
    }

    m_stats.inc_stats(mf->get_access_type(), m_stats.select_stats_status(status, cache_status));
    return cache_status;
}

//! A general function that takes the result of a tag_array probe
//  and performs the correspding functions based on the cache configuration
//  The access fucntion calls this function
enum cache_request_status
data_cache::process_tag_probe( bool wr,
                               enum cache_request_status probe_status,
                               new_addr_type addr,
                               unsigned cache_index,
                               mem_fetch* mf,
                               unsigned time,
                               std::list<cache_event>& events )
{
    // Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the
    // data_cache constructor to reflect the corresponding cache configuration
    // options. Function pointers were used to avoid many long conditional
    // branches resulting from many cache configuration options.
    cache_request_status access_status = probe_status;
    if(wr){ // Write
        if(probe_status == HIT){
            access_status = (this->*m_wr_hit)( addr,
                                      cache_index,
                                      mf, time, events, probe_status );
        }else if ( probe_status != RESERVATION_FAIL ) {
            access_status = (this->*m_wr_miss)( addr,
                                       cache_index,
                                       mf, time, events, probe_status );
        }
    }else{ // Read
        if(probe_status == HIT){
            access_status = (this->*m_rd_hit)( addr,
                                      cache_index,
                                      mf, time, events, probe_status );
        }else if ( probe_status != RESERVATION_FAIL ) {
            access_status = (this->*m_rd_miss)( addr,
                                       cache_index,
                                       mf, time, events, probe_status );
        }
    }

    m_bandwidth_management.use_data_port(mf, access_status, events); 
    return access_status;
}

// Both the L1 and L2 currently use the same access function.
// Differentiation between the two caches is done through configuration
// of caching policies.
// Both the L1 and L2 override this function to provide a means of
// performing actions specific to each cache when such actions are implemnted.
enum cache_request_status
data_cache::access( new_addr_type addr,
                    mem_fetch *mf,
                    unsigned time,
                    std::list<cache_event> &events )
{
    //if(m_config.m_enable_prefetch && !m_extra_accessq.empty()) {
    //    spare_cycle(time);
    //    return RESERVATION_FAIL;
    //}

    assert( mf->get_data_size() <= m_config.get_line_sz());
    bool wr = mf->get_is_write();
    new_addr_type block_addr = m_config.block_addr(addr);
    unsigned cache_index = (unsigned)-1;
    enum cache_request_status probe_status
        = m_tag_array->probe( mf, block_addr, cache_index );
    enum cache_request_status access_status
        = process_tag_probe( wr, probe_status, addr, cache_index, mf, time, events );
    m_stats.inc_stats(mf->get_access_type(),
        m_stats.select_stats_status(probe_status, access_status));

    if(m_config.m_enable_prefetch) {
        prefetch(mf, time, probe_status);
        //spare_cycle(time);
    }
    return access_status;
}

/// This is meant to model the first level data cache in Fermi.
/// It is write-evict (global) or write-back (local) at the
/// granularity of individual blocks (Set by GPGPU-Sim configuration file)
/// (the policy used in fermi according to the CUDA manual)
enum cache_request_status
l1_cache::access( new_addr_type addr,
                  mem_fetch *mf,
                  unsigned time,
                  std::list<cache_event> &events )
{
    return data_cache::access( addr, mf, time, events );
}

// The l2 cache access function calls the base data_cache access
// implementation.  When the L2 needs to diverge from L1, L2 specific
// changes should be made here.
enum cache_request_status
l2_cache::access( new_addr_type addr,
                  mem_fetch *mf,
                  unsigned time,
                  std::list<cache_event> &events )
{
    return data_cache::access( addr, mf, time, events );
}

// lld: spare cycles
void data_cache::spare_cycle( unsigned time )
{
    // issue prefetch requests
    if(!m_extra_accessq.empty()) {
        std::list<cache_event> events;
        mem_fetch *mf = m_extra_accessq.front();
        new_addr_type addr = mf->get_addr();

        assert( mf->get_data_size() <= m_config.get_line_sz());
        bool wr = mf->get_is_write();
        new_addr_type block_addr = m_config.block_addr(addr);
        unsigned cache_index = (unsigned)-1;
        enum cache_request_status probe_status
            = m_tag_array->probe( mf, block_addr, cache_index );
        enum cache_request_status access_status = probe_status;
        if(probe_status == HIT || probe_status == HIT_RESERVED) {
            m_extra_accessq.pop_front();
            delete mf;
        } else if(probe_status == MISS) {
            access_status = process_tag_probe( wr, probe_status, addr, cache_index, mf, time, events );
            if(access_status != RESERVATION_FAIL) {
                m_extra_accessq.pop_front();
                if(access_status == HIT || access_status == HIT_RESERVED) // it means prefetch is bypassed
                    delete mf;
            }
        }
        m_stats.inc_stats(mf->get_access_type(),
            m_stats.select_stats_status(probe_status, access_status));
    }
}

void l1_cache::spare_cycle( unsigned time )
{
    data_cache::spare_cycle(time);
}

void l2_cache::spare_cycle( unsigned time )
{
    data_cache::spare_cycle(time);
}

// lld: generate prefetch requests
void data_cache::prefetch( mem_fetch *mf, unsigned time, enum cache_request_status probe_status )
{
    //if(m_extra_accessq.size() < m_config.m_extra_accessq_size &&
    //   (probe_status == MISS || probe_status == RESERVATION_FAIL)) {
    if(m_extra_accessq.size() < m_config.m_extra_accessq_size) {
        new_addr_type pref_addr = m_config.block_addr(mf->get_addr()) + m_config.get_line_sz();
        // check redundant prefetch request
        for(list<mem_fetch*>::iterator it = m_extra_accessq.begin(); it != m_extra_accessq.end(); it++) {
            if((*it)->get_addr() == pref_addr)
                return;
        }
        // check if this is a demand request by this instruction
        if(mf->get_inst().accessq_find(pref_addr))
            return;
        mem_fetch *pref_mf = m_memfetch_creator->alloc(pref_addr, m_l2?L2_PREF_ACC_R:L1_PREF_ACC_R, m_config.get_line_sz(), false);
        pref_mf->set_prefetch_info(m_l2, mf->get_pc(), 1);
        m_extra_accessq.push_back(pref_mf);
    }
}

/// Access function for tex_cache
/// return values: RESERVATION_FAIL if request could not be accepted
/// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
/// since unlike a normal CPU cache, a "HIT" in texture cache does not
/// mean the data is ready (still need to get through fragment fifo)
enum cache_request_status tex_cache::access( new_addr_type addr, mem_fetch *mf,
    unsigned time, std::list<cache_event> &events )
{
    if ( m_fragment_fifo.full() || m_request_fifo.full() || m_rob.full() )
        return RESERVATION_FAIL;

    assert( mf->get_data_size() <= m_config.get_line_sz());

    // at this point, we will accept the request : access tags and immediately allocate line
    new_addr_type block_addr = m_config.block_addr(addr);
    unsigned cache_index = (unsigned)-1;
    enum cache_request_status status = m_tags.access(mf,block_addr,time,cache_index);
    enum cache_request_status cache_status = RESERVATION_FAIL;
    assert( status != RESERVATION_FAIL );
    assert( status != HIT_RESERVED ); // as far as tags are concerned: HIT or MISS
    m_fragment_fifo.push( fragment_entry(mf,cache_index,status==MISS,mf->get_data_size()) );
    if ( status == MISS ) {
        // we need to send a memory request...
        unsigned rob_index = m_rob.push( rob_entry(cache_index, mf, block_addr) );
        m_extra_mf_fields[mf] = extra_mf_fields(rob_index);
        mf->set_data_size(m_config.get_line_sz());
        m_tags.fill(mf,block_addr,cache_index,time); // mark block as valid
        m_request_fifo.push(mf);
        mf->set_status(m_request_queue_status,time);
        events.push_back(READ_REQUEST_SENT);
        cache_status = MISS;
    } else {
        // the value *will* *be* in the cache already
        cache_status = HIT_RESERVED;
    }
    m_stats.inc_stats(mf->get_access_type(), m_stats.select_stats_status(status, cache_status));
    return cache_status;
}

void tex_cache::cycle(){
    // send next request to lower level of memory
    if ( !m_request_fifo.empty() ) {
        mem_fetch *mf = m_request_fifo.peek();
        if ( !m_memport->full(mf->get_ctrl_size(),false) ) {
            m_request_fifo.pop();
            m_memport->push(mf);
        }
    }
    // read ready lines from cache
    if ( !m_fragment_fifo.empty() && !m_result_fifo.full() ) {
        const fragment_entry &e = m_fragment_fifo.peek();
        if ( e.m_miss ) {
            // check head of reorder buffer to see if data is back from memory
            unsigned rob_index = m_rob.next_pop_index();
            const rob_entry &r = m_rob.peek(rob_index);
            assert( r.m_request == e.m_request );
            assert( r.m_block_addr == m_config.block_addr(e.m_request->get_addr()) );
            if ( r.m_ready ) {
                assert( r.m_index == e.m_cache_index );
                m_cache[r.m_index].m_valid = true;
                m_cache[r.m_index].m_block_addr = r.m_block_addr;
                m_result_fifo.push(e.m_request);
                m_rob.pop();
                m_fragment_fifo.pop();
            }
        } else {
            // hit:
            assert( m_cache[e.m_cache_index].m_valid );
            assert( m_cache[e.m_cache_index].m_block_addr
                == m_config.block_addr(e.m_request->get_addr()) );
            m_result_fifo.push( e.m_request );
            m_fragment_fifo.pop();
        }
    }
}

/// Place returning cache block into reorder buffer
void tex_cache::fill( mem_fetch *mf, unsigned time )
{
    extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
    assert( e != m_extra_mf_fields.end() );
    assert( e->second.m_valid );
    assert( !m_rob.empty() );
    mf->set_status(m_rob_status,time);

    unsigned rob_index = e->second.m_rob_index;
    rob_entry &r = m_rob.peek(rob_index);
    assert( !r.m_ready );
    r.m_ready = true;
    r.m_time = time;
    assert( r.m_block_addr == m_config.block_addr(mf->get_addr()) );
}

void tex_cache::display_state( FILE *fp ) const
{
    fprintf(fp,"%s (texture cache) state:\n", m_name.c_str() );
    fprintf(fp,"fragment fifo entries  = %u / %u\n",
        m_fragment_fifo.size(), m_fragment_fifo.capacity() );
    fprintf(fp,"reorder buffer entries = %u / %u\n",
        m_rob.size(), m_rob.capacity() );
    fprintf(fp,"request fifo entries   = %u / %u\n",
        m_request_fifo.size(), m_request_fifo.capacity() );
    if ( !m_rob.empty() )
        fprintf(fp,"reorder buffer contents:\n");
    for ( int n=m_rob.size()-1; n>=0; n-- ) {
        unsigned index = (m_rob.next_pop_index() + n)%m_rob.capacity();
        const rob_entry &r = m_rob.peek(index);
        fprintf(fp, "tex rob[%3d] : %s ",
            index, (r.m_ready?"ready  ":"pending") );
        if ( r.m_ready )
            fprintf(fp,"@%6u", r.m_time );
        else
            fprintf(fp,"       ");
        fprintf(fp,"[idx=%4u]",r.m_index);
        r.m_request->print(fp,false);
    }
    if ( !m_fragment_fifo.empty() ) {
        fprintf(fp,"fragment fifo (oldest) :");
        fragment_entry &f = m_fragment_fifo.peek();
        fprintf(fp,"%s:          ", f.m_miss?"miss":"hit ");
        f.m_request->print(fp,false);
    }
}

enum cache_request_status tag_array::probe_selector( mem_fetch *mf, new_addr_type addr, unsigned &idx, std::list<unsigned> &repl_set, unsigned time )
{
    //if( g_debug_execution >= 1 && m_core_id == 0 && m_config.m_custom_repl == 1 ) {
    if( g_debug_execution >= 1 && m_config.m_custom_repl == 1 ) {
        cerr << dec << "Core " << m_core_id << hex << " accesses 0x" << addr << dec << " (No. " << m_access << ")" << endl;
        //cerr << "PHit: " << m_pending_hit << ", Miss: " << m_miss << endl;
    }
#ifdef PRINT_TRACE
    if(m_core_id == 0 && m_type_id == 0) {
        trace_file << hex << "0x" << addr << " ";
        for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++)
            trace_file << mf->get_original_sector_mask(m_core_id<0).test(i);
        trace_file << endl;
    }
#endif

    if( m_config.m_alloc_policy == ON_MISS )
        return access_probe( mf, addr, idx, repl_set, time );
    else if (  m_config.m_alloc_policy == ON_FILL )
        return probe( mf, addr, idx );
    else abort();
}

enum cache_request_status tag_array::access_probe( mem_fetch *mf, new_addr_type addr, unsigned &idx, std::list<unsigned> &repl_set, unsigned time )
{
    //assert( m_config.m_write_policy == READ_ONLY );
    unsigned set_index = m_config.set_index(addr);
    new_addr_type tag = m_config.tag(addr);

    unsigned invalid_line = (unsigned)-1;
    unsigned valid_line = (unsigned)-1;
    unsigned valid_timestamp = (unsigned)-1;

    bool all_reserved = true;
    bool all_used = true; // lld
    bool have_valid = false; // lld
    unsigned unused_line = (unsigned)-1; // lld
    idx = (unsigned)-1; // lld

    // check for hit or pending hit
    cache_request_status status = MISS;
    for (unsigned way=0; way<m_config.m_assoc; way++) {
        unsigned index = set_index*m_config.m_assoc+way;
        cache_block_t *line = &m_lines[index];
        if (line->m_tag == tag && line->m_status != INVALID && line->m_status != UNUSED && status == MISS) { // lld
            // lld: sector cache
            if( m_config.m_replacement_policy == SECTOR || m_config.m_replacement_policy == ADAP_GRAN )
            {
                idx = index;
                if ( line->m_status == RESERVED )
                    status = HIT_RESERVED;
                else if ( line->m_status == VALID )
                    status = HIT;
                else if ( line->m_status == MODIFIED )
                    status = HIT;
                else
                    assert(0);

                for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++)
                    if(mf->get_original_sector_mask(m_core_id<0).test(i) && !line->m_sector_mask.test(i)) // lld: sector and adaptive grain cache
                        status = MISS_PARTIAL;
                if(status != MISS_PARTIAL) {
                    for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++)
                        if(mf->get_original_sector_mask(m_core_id<0).test(i) && !line->m_ready_sector_mask.test(i)) // lld: sector and adaptive grain cache
                            status = HIT_RESERVED;
                }
                if(status == HIT || status == HIT_RESERVED)
                    return status;
            }
            else
            {
                if ( line->m_status == RESERVED ) {
                    idx = index;
                    return HIT_RESERVED;
                } else if ( line->m_status == VALID ) {
                    idx = index;
                    return HIT;
                } else if ( line->m_status == MODIFIED ) {
                    idx = index;
                    return HIT;
                } else {
                    assert( line->m_status == INVALID );
                }
            }
        }
        if (line->m_status != RESERVED && line->m_status != UNUSED && index != idx) { // lld: cannot replace itself
            all_reserved = false;
            if (line->m_status == INVALID) {
                invalid_line = index;
            } else {
                // valid line : keep track of most appropriate replacement candidate
                if ( m_config.m_replacement_policy == LRU || m_config.m_custom_repl == 0 ) { // lld: use lru replacement
                    if ( line->m_last_access_time < valid_timestamp ) {
                        valid_timestamp = line->m_last_access_time;
                        valid_line = index;
                    }
                } else if ( m_config.m_replacement_policy == FIFO ) {
                    if ( line->m_alloc_time < valid_timestamp ) {
                        valid_timestamp = line->m_alloc_time;
                        valid_line = index;
                    }
                }
            }
        }
        if(line->m_status == UNUSED) { // lld
            all_used = false;
            unused_line = index;
        }
        if((line->m_status == VALID || line->m_status == MODIFIED) && index != idx) // lld: check if there is anything replaceable
            have_valid = true;
        if(line->m_status == RESERVED && time - line->m_warn_time >= 10000) { // lld: check if it is reserved for a long time
            cerr << "WARN: line " << index << " (addr: " << line->m_block_addr << ") in cache (" << m_core_id << ", "
                 << m_type_id << ") has been reserved from " << line->m_alloc_time << " to " << time << "." << endl;
            line->m_warn_time = time;
            //assert(0);
        }
    }

    // lld: potentially replace on MISS_PARTIAL
    update_sets(set_index);
    if(m_config.m_replacement_policy == ADAP_GRAN && status == MISS_PARTIAL) {
        unsigned extra_size = 0;
        unsigned ori_extra_size = 0;
        for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++) {
            if(mf->get_alloc_sector_mask(m_core_id<0).test(i) && !m_lines[idx].m_sector_mask.test(i))
                extra_size++;
            if(mf->get_original_sector_mask(m_core_id<0).test(i) && !m_lines[idx].m_sector_mask.test(i))
                ori_extra_size++;
        }

        if(m_sets[set_index].m_size + extra_size <= m_config.m_assoc) {
            if ( m_config.m_custom_repl == 1 && have_valid )
                if(m_sets[set_index].m_size + ori_extra_size > m_config.m_assoc)
                    get_partial_victim(mf, addr, idx); // lld: just for updating BM
            return status;
        }

        if ( !have_valid ) {
            assert( m_config.m_alloc_policy == ON_MISS ); 
            return RESERVATION_FAIL; // miss and not enough space in cache to allocate on miss
        }
        unsigned repl_idx;
        if ( m_config.m_custom_repl == 1 ) {
            if(mf->get_done(m_core_id<0))
                repl_idx = get_partial_victim(mf, addr, idx); // lld: cannot replace itself
            else // lld: cannot bypass for the rest replacement
                repl_idx = get_real_victim(mf, addr, idx);
        } else if ( valid_line != (unsigned)-1) {
            repl_idx = valid_line;
        } else abort(); // if an unreserved block exists, it is either invalid or replaceable 

        repl_set.push_back(repl_idx);
        return status;
    } else if(m_config.m_replacement_policy == SECTOR && status == MISS_PARTIAL)
        return status;

    bool need_real_repl = false;
    if( m_config.m_replacement_policy == ADAP_GRAN ) {
        assert(status == MISS);
        if(m_sets[set_index].m_size + mf->get_alloc_sector_mask(m_core_id<0).count() > m_config.m_assoc) {
            need_real_repl = true;
            invalid_line = (unsigned)-1;
        } else if(mf->get_alloc_sector_mask(m_core_id<0).count() == 0) { // lld: equal to bypass
            if ( m_config.m_custom_repl == 1 && have_valid )
                if(m_sets[set_index].m_size + mf->get_original_sector_mask(m_core_id<0).count() > m_config.m_assoc)
                    get_victim(mf, addr); // lld: just for updating BM
            idx = (unsigned)-1;
            return MISS;
        } else if(!all_used) {
            if ( invalid_line != (unsigned)-1 )
                idx = invalid_line;
            else
                idx = unused_line;
            return MISS;
        }
    }
    if(need_real_repl && !have_valid)
        return RESERVATION_FAIL;
    if ( all_reserved ) {
        assert( m_config.m_alloc_policy == ON_MISS ); 
        return RESERVATION_FAIL; // miss and not enough space in cache to allocate on miss
    }
    if( m_config.m_replacement_policy == PARTITION && !check_avail(mf, addr) ) // lld: check reservation fail for partition
        return RESERVATION_FAIL;

    if ( invalid_line != (unsigned)-1 ) {
        idx = invalid_line;
    } else if ( m_config.m_custom_repl == 1 ) {
        //idx = get_victim(mf, addr);
        if(mf->get_done(m_core_id<0))
            idx = get_victim(mf, addr);
        else // lld: cannot bypass for the rest replacement
            idx = get_real_victim(mf, addr, (unsigned)-1);
    } else if ( valid_line != (unsigned)-1) {
        idx = valid_line;
    } else abort(); // if an unreserved block exists, it is either invalid or replaceable 

    return MISS;
}

enum cache_request_status tag_array::fill_probe( mem_fetch *mf, new_addr_type addr, unsigned &idx ) {
    //assert( m_config.m_write_policy == READ_ONLY );
    unsigned set_index = m_config.set_index(addr);
    new_addr_type tag = m_config.tag(addr);

    /*
    unsigned invalid_line = (unsigned)-1;
    unsigned valid_line = (unsigned)-1;
    unsigned valid_timestamp = (unsigned)-1;
    */

    bool all_reserved = true;
    bool all_used = true; // lld
    bool have_valid = false; // lld
    idx = (unsigned)-1; // lld

    // check for hit or pending hit
    cache_request_status status = MISS;
    for (unsigned way=0; way<m_config.m_assoc; way++) {
        unsigned index = set_index*m_config.m_assoc+way;
        cache_block_t *line = &m_lines[index];
        if (line->m_tag == tag && line->m_status != INVALID && line->m_status != UNUSED && status == MISS) { // lld
            // lld: sector cache
            if( m_config.m_replacement_policy == SECTOR || m_config.m_replacement_policy == ADAP_GRAN )
            {
                idx = index;
                if ( line->m_status == RESERVED )
                    status = HIT_RESERVED;
                else if ( line->m_status == VALID )
                    status = HIT;
                else if ( line->m_status == MODIFIED )
                    status = HIT;
                else
                    assert(0);

                for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++)
                    if(mf->get_original_sector_mask(m_core_id<0).test(i) && !line->m_sector_mask.test(i)) // lld: sector and adaptive grain cache
                        return MISS_PARTIAL;
                for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++)
                    if(mf->get_original_sector_mask(m_core_id<0).test(i) && !line->m_ready_sector_mask.test(i)) // lld: sector and adaptive grain cache
                        return HIT_RESERVED;
                return HIT;
            }
            else
            {
                if ( line->m_status == RESERVED ) {
                    idx = index;
                    return HIT_RESERVED;
                } else if ( line->m_status == VALID ) {
                    idx = index;
                    return HIT;
                } else if ( line->m_status == MODIFIED ) {
                    idx = index;
                    return HIT;
                } else {
                    assert( line->m_status == INVALID );
                }
            }
        }
        if (line->m_status != RESERVED && line->m_status != UNUSED) { // lld
            all_reserved = false;
            /*
            if (line->m_status == INVALID) {
                invalid_line = index;
            } else {
                // valid line : keep track of most appropriate replacement candidate
                if ( m_config.m_replacement_policy == LRU ) {
                    if ( line->m_last_access_time < valid_timestamp ) {
                        valid_timestamp = line->m_last_access_time;
                        valid_line = index;
                    }
                } else if ( m_config.m_replacement_policy == FIFO ) {
                    if ( line->m_alloc_time < valid_timestamp ) {
                        valid_timestamp = line->m_alloc_time;
                        valid_line = index;
                    }
                }
            }
            */
        }
        if(line->m_status == UNUSED) // lld
            all_used = false;
        if((line->m_status == VALID || line->m_status == MODIFIED) && index != idx) // lld
            have_valid = true;
    }

    return MISS;
}

void tag_array::update_stat(mem_fetch *mf, unsigned idx, bool hit)
{
    if(hit)
    {
        // inter or intra hit
        if(m_lines[idx].m_wid == mf->get_wid()) m_intra_warp_hit++;
        else m_inter_warp_hit++;
        m_lines[idx].m_wid = mf->get_wid();

        // reuse within cache line
        bitset<MAX_CACHE_CHUNK_NUM> chunk_mask;
        int cont_chunk_size = 0;
        int chunk_start = -1, chunk_end = -1;
        chunk_mask.reset();
        for(unsigned i = 0; i < MAX_CACHE_CHUNK_NUM; i++)
        {
            for(unsigned j = 0; j < CACHE_CHUNK_SIZE; j++)
                if(m_lines[idx].m_reuse.test(i*CACHE_CHUNK_SIZE+j))
                {
                    chunk_mask.set(i);
                    if(chunk_start == -1)
                        chunk_start = i;
                    break;
                }
        }
        assert(m_chunk_reuse[chunk_mask.count()] != 0);
        m_chunk_reuse[chunk_mask.count()]--;
        for(int i = MAX_CACHE_CHUNK_NUM-1; i >= 0; i--)
            if(chunk_mask.test(i))
            {
                chunk_end = i;
                break;
            }
        if(chunk_end >= 0)
        {
            cont_chunk_size = chunk_end - chunk_start + 1;
            assert(cont_chunk_size > 0);
        }
        assert(cont_chunk_size >= (int)chunk_mask.count());
        assert(m_chunk_cont_reuse[cont_chunk_size]);
        m_chunk_cont_reuse[cont_chunk_size]--;
        assert(m_reuse[m_lines[idx].m_reuse.count()] != 0);
        m_reuse[m_lines[idx].m_reuse.count()]--;

        m_lines[idx].m_reuse |= mf->get_access_byte_mask();

        cont_chunk_size = 0;
        chunk_start = -1;
        chunk_end = -1;
        chunk_mask.reset();
        for(unsigned i = 0; i < MAX_CACHE_CHUNK_NUM; i++)
        {
            for(unsigned j = 0; j < CACHE_CHUNK_SIZE; j++)
                if(m_lines[idx].m_reuse.test(i*CACHE_CHUNK_SIZE+j))
                {
                    chunk_mask.set(i);
                    if(chunk_start == -1)
                        chunk_start = i;
                    break;
                }
        }
        m_chunk_reuse[chunk_mask.count()]++;
        for(int i = MAX_CACHE_CHUNK_NUM-1; i >= 0; i--)
            if(chunk_mask.test(i))
            {
                chunk_end = i;
                break;
            }
        if(chunk_end >= 0)
        {
            cont_chunk_size = chunk_end - chunk_start + 1;
            assert(cont_chunk_size > 0);
        }
        assert(cont_chunk_size >= (int)chunk_mask.count());
        m_chunk_cont_reuse[cont_chunk_size]++;
        m_reuse[m_lines[idx].m_reuse.count()]++;

        // useful prefetch
        if(m_lines[idx].m_prefetch) {
            m_lines[idx].m_prefetch = false;
            m_useful_prefetch++;
        } else {
            for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++) // lld: sector and adaptive grain cache
                if(mf->get_original_sector_mask(m_core_id<0).test(i) && m_lines[idx].m_sector_mask.test(i) && !m_lines[idx].m_original_sector_mask.test(i)) {
                    m_lines[idx].m_original_sector_mask.set(i);
                    m_lines[idx].m_pref_sector_mask.reset(i);
                    m_useful_prefetch++;
                }
        }
    }
    else
    {
        if(mf->get_done(m_core_id<0)) {
#ifndef COLLECT_INNER_USE
            m_reuse[0]++;
            m_chunk_reuse[0]++;
            m_chunk_cont_reuse[0]++;
#else
            // lld: for use information
            m_lines[idx].m_reuse = mf->get_access_byte_mask();
            int cont_chunk_size = 0;
            int chunk_start = -1;
            int chunk_end = -1;
            bitset<MAX_CACHE_CHUNK_NUM> chunk_mask;
            chunk_mask.reset();
            for(unsigned i = 0; i < MAX_CACHE_CHUNK_NUM; i++)
            {
                for(unsigned j = 0; j < CACHE_CHUNK_SIZE; j++)
                    if(m_lines[idx].m_reuse.test(i*CACHE_CHUNK_SIZE+j))
                    {
                        chunk_mask.set(i);
                        if(chunk_start == -1)
                            chunk_start = i;
                        break;
                    }
            }
            m_chunk_reuse[chunk_mask.count()]++;
            for(int i = MAX_CACHE_CHUNK_NUM-1; i >= 0; i--)
                if(chunk_mask.test(i))
                {
                    chunk_end = i;
                    break;
                }
            if(chunk_end >= 0)
            {
                cont_chunk_size = chunk_end - chunk_start + 1;
                assert(cont_chunk_size > 0);
            }
            assert(cont_chunk_size >= (int)chunk_mask.count());
            m_chunk_cont_reuse[cont_chunk_size]++;
            m_reuse[m_lines[idx].m_reuse.count()]++;
#endif
        }

        if(m_lines[idx].m_prefetch)
            m_useless_prefetch++;
        else {
            if(m_lines[idx].m_status == VALID || m_lines[idx].m_status == MODIFIED)
                for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++) // lld: sector and adaptive grain cache
                    if(m_lines[idx].m_sector_mask.test(i) && !m_lines[idx].m_original_sector_mask.test(i))
                        m_useless_prefetch++;
        }
    }
}

// lld: update set status
void tag_array::update_sets(unsigned set_index)
{
    if(m_config.m_replacement_policy != ADAP_GRAN)
        return;

    m_sets[set_index].m_size = 0;
    for (unsigned way=0; way<m_config.m_assoc; way++) {
        unsigned index = set_index*m_config.m_assoc+way;
        cache_block_t *line = &m_lines[index];
        if (line->m_status != INVALID && line->m_status != UNUSED)
            m_sets[set_index].m_size += line->m_sector_mask.count();
    }
    assert(m_sets[set_index].m_size <= m_config.m_assoc);
}

const char * cache_repl_stats_str(enum cache_repl_stats id) 
{
   #define CRS_TUP_BEGIN(X) static const char* access_type_str[] = {
   #define CRS_TUP(X) #X
   #define CRS_TUP_END(X) };
   CACHE_REPL_STATS_TUP_DEF
   #undef CRS_TUP_BEGIN
   #undef CRS_TUP
   #undef CRS_TUP_END

   assert(id < NUM_CACHE_REPL_STATS); 
   return access_type_str[id]; 
}

void baseline_cache::issue_request(new_addr_type block_addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, bool wa)
{
    assert(mf->get_type() == READ_REQUEST);
    bool found = false;
    if(m_config.m_mshr_type == LARGE_MSHR) {
        // look for request in miss queue
        new_addr_type mshr_block_addr = m_config.coalesced_block_addr(mf->get_addr());
        for(list<mem_fetch*>::iterator it = m_miss_queue.begin(); it != m_miss_queue.end(); it++) {
            if(mshr_block_addr == m_config.coalesced_block_addr((*it)->get_addr()) && (*it)->get_type() == READ_REQUEST) {
                found = true;
                if((*it)->get_data_size() == m_config.get_line_sz()) {
                    // transfer this request to a larger one
                    new_addr_type ori_block_addr = m_config.block_addr((*it)->get_addr());
                    assert(block_addr != ori_block_addr);
                    (*it)->set_addr(mshr_block_addr);
                    for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++)
                        (*it)->set_sector_mask(m_l2, i, false);
                    assert((*it)->get_sector_mask(m_l2).count() == 0);
                    for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++)
                        (*it)->set_sector_mask(m_l2, (ori_block_addr-mshr_block_addr)/CACHE_CHUNK_SIZE+i, true);
                    (*it)->set_original_sector_mask(m_l2);
                } else
                    assert((*it)->get_addr() == mshr_block_addr);
                // update the sector mask of issued request
                for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++)
                    (*it)->set_sector_mask(m_l2, (block_addr-mshr_block_addr)/CACHE_CHUNK_SIZE+i, true);
                (*it)->set_data_size( (*it)->get_sector_mask(m_l2).count()*CACHE_CHUNK_SIZE );
                (*it)->inc_combination(m_l2);
                assert((*it)->get_sector_mask(m_l2).count() == ((*it)->get_combination(m_l2)+1)*m_config.get_line_sz()/CACHE_CHUNK_SIZE);
                (*it)->m_merged_requests.push_back(mf);
                assert((*it)->m_merged_requests.size() == (*it)->get_combination(m_l2));
                break;
            }
        }
    }

    m_extra_mf_fields[mf] = extra_mf_fields(block_addr,cache_index,mf->get_data_size());
    if(mf->get_prefetch(m_l2))
        m_extra_pending_list.push_back(mf);
    if(!found) {
        if(m_config.m_mshr_type == SECTOR_MSHR || m_config.m_mshr_type == LARGE_MSHR) // lld: sector cache
            mf->set_data_size( mf->get_sector_mask(m_l2).count()*CACHE_CHUNK_SIZE );
        else
            mf->set_data_size( m_config.get_line_sz() );
        m_miss_queue.push_back(mf);
        mf->set_issue(m_l2);
        mf->set_issue_time(m_l2, time); // lld: for memory latency calculation
        mf->set_status(m_miss_queue_status,time);
        if(!wa)
            events.push_back(READ_REQUEST_SENT);
    }
}
/******************************************************************************************************************************************/

