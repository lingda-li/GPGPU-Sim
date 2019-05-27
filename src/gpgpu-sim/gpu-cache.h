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

#ifndef GPU_CACHE_H
#define GPU_CACHE_H

#include <stdio.h>
#include <stdlib.h>
#include "gpu-misc.h"
#include "mem_fetch.h"
#include "../abstract_hardware_model.h"
#include "../tr1_hash_map.h"

#include "addrdec.h"

enum cache_block_state {
    INVALID,
    RESERVED,
    VALID,
    MODIFIED,
    UNUSED // lld: currently unused block, cannot be used by any line
};

enum cache_event {
    WRITE_BACK_REQUEST_SENT,
    READ_REQUEST_SENT,
    WRITE_REQUEST_SENT
};

const char * cache_request_status_str(enum cache_request_status status); 

struct cache_block_t {
    cache_block_t()
    {
        m_tag=0;
        m_block_addr=0;
        m_alloc_time=0;
        m_fill_time=0;
        m_last_access_time=0;
        m_status=INVALID;
        // lld
        m_prefetch = false;
        m_sector_mask.reset();
        m_ready_sector_mask.reset();
        m_original_sector_mask.reset();
        m_pref_sector_mask.reset();
        m_wid = 0;
        m_reuse.reset();
        m_size = 0;
        m_warn_time = 0;
    }
    void allocate( new_addr_type tag, new_addr_type block_addr, unsigned time, unsigned wid )
    {
        m_tag=tag;
        m_block_addr=block_addr;
        m_alloc_time=time;
        m_last_access_time=time;
        m_fill_time=0;
        m_status=RESERVED;
        // lld
        m_prefetch = false;
        m_sector_mask.reset();
        m_ready_sector_mask.reset();
        m_original_sector_mask.reset();
        m_pref_sector_mask.reset();
        m_wid = wid;
        m_reuse.reset();
        m_size = 0;
        m_warn_time = time;
    }
    void fill( unsigned time, unsigned wid )
    {
        assert( m_status == RESERVED );
        m_status=VALID;
        m_fill_time=time;
    }

    new_addr_type    m_tag;
    new_addr_type    m_block_addr;
    unsigned         m_alloc_time;
    unsigned         m_last_access_time;
    unsigned         m_fill_time;
    cache_block_state    m_status;

    // lld
    bool m_prefetch;
    std::bitset<MAX_CACHE_CHUNK_NUM> m_sector_mask;
    std::bitset<MAX_CACHE_CHUNK_NUM> m_ready_sector_mask;
    std::bitset<MAX_CACHE_CHUNK_NUM> m_original_sector_mask;
    std::bitset<MAX_CACHE_CHUNK_NUM> m_pref_sector_mask;
    unsigned m_size;
    unsigned m_wid;
    mem_access_byte_mask_t m_reuse;
    unsigned         m_warn_time;
};

// lld: cache set
struct cache_set_t {
    ~cache_set_t()
    {
        delete[] m_idx;
    }
    void init(unsigned max_line)
    {
        m_line_num = 0;
        m_size = 0;
        m_idx = new unsigned[max_line];
        for(unsigned i = 0; i < max_line; i++)
            m_idx[i] = (unsigned)-1;
    }
    void reset(unsigned max_line)
    {
        m_line_num = 0;
        m_size = 0;
        for(unsigned i = 0; i < max_line; i++)
            m_idx[i] = (unsigned)-1;
    }

    unsigned m_line_num;
    unsigned m_size; // number of chunk
    unsigned *m_idx;
};

enum replacement_policy_t {
    LRU,
    FIFO,
    MY_REPL,
    SECTOR,
    ADAP_GRAN,
    PREF,
    PARTITION
};

enum write_policy_t {
    READ_ONLY,
    WRITE_BACK,
    WRITE_THROUGH,
    WRITE_EVICT,
    LOCAL_WB_GLOBAL_WT
};

enum allocation_policy_t {
    ON_MISS,
    ON_FILL
};


enum write_allocate_policy_t {
	NO_WRITE_ALLOCATE,
	WRITE_ALLOCATE
};

enum mshr_config_t {
    TEX_FIFO,
    ASSOC, // normal cache 
    SECTOR_MSHR, // lld: sector cache mshr
    LARGE_MSHR // lld: large granularity mshr
};

enum set_index_function{
    FERMI_HASH_SET_FUNCTION = 0,
    LINEAR_SET_FUNCTION,
    CUSTOM_SET_FUNCTION
};

class cache_config {
public:
    cache_config() 
    { 
        m_valid = false; 
        m_disabled = false;
        m_config_string = NULL; // set by option parser
        m_config_stringPrefL1 = NULL;
        m_config_stringPrefShared = NULL;
        m_data_port_width = 0;
        m_set_index_function = LINEAR_SET_FUNCTION;
    }
    void init(char * config, FuncCache status)
    {
    	cache_status= status;
        assert( config );
        char rp, wp, ap, mshr_type, wap, sif;


        int ntok = sscanf(config,"%u:%u:%u,%c:%c:%c:%c:%c,%c:%u:%u,%u:%u,%u",
                          &m_nset, &m_line_sz, &m_assoc, &rp, &wp, &ap, &wap,
                          &sif,&mshr_type,&m_mshr_entries,&m_mshr_max_merge,
                          &m_miss_queue_size, &m_result_fifo_entries,
                          &m_data_port_width);

        m_true_assoc = m_assoc; // lld: adaptive granularity
        if ( ntok < 11 ) {
            if ( !strcmp(config,"none") ) {
                m_disabled = true;
                return;
            }
            exit_parse_error();
        }
        m_enable_prefetch = false;
        switch (rp) {
        case 'L': m_replacement_policy = LRU; m_custom_repl=2; break;
        case 'F': m_replacement_policy = FIFO; m_custom_repl=2; break;
        case 'M': m_replacement_policy = MY_REPL; m_custom_repl=1; break;
        case 'S': m_replacement_policy = SECTOR; m_custom_repl=0; break;
        case 'G': m_replacement_policy = ADAP_GRAN; m_custom_repl=1; m_assoc*=m_line_sz/CACHE_CHUNK_SIZE; break;
        case 'P': m_replacement_policy = PREF; m_custom_repl=1; m_enable_prefetch=true; break;
        case 'T': m_replacement_policy = PARTITION; m_custom_repl=1; break;
        default: exit_parse_error();
        }
        switch (wp) {
        case 'R': m_write_policy = READ_ONLY; break;
        case 'B': m_write_policy = WRITE_BACK; break;
        case 'T': m_write_policy = WRITE_THROUGH; break;
        case 'E': m_write_policy = WRITE_EVICT; break;
        case 'L': m_write_policy = LOCAL_WB_GLOBAL_WT; break;
        default: exit_parse_error();
        }
        switch (ap) {
        case 'm': m_alloc_policy = ON_MISS; break;
        case 'f': m_alloc_policy = ON_FILL; break;
        default: exit_parse_error();
        }
        switch (mshr_type) {
        case 'F': m_mshr_type = TEX_FIFO; assert(ntok==13); break;
        case 'A':
            if(m_replacement_policy == SECTOR)
                m_mshr_type = SECTOR_MSHR;
            else if(m_replacement_policy == ADAP_GRAN)
                //m_mshr_type = LARGE_MSHR;
                m_mshr_type = SECTOR_MSHR;
            else if(m_replacement_policy == PREF)
                m_mshr_type = LARGE_MSHR;
            else
                m_mshr_type = ASSOC;
            break;
        default: exit_parse_error();
        }
        m_line_sz_log2 = LOGB2(m_line_sz);
        m_nset_log2 = LOGB2(m_nset);
        m_valid = true;

        switch(wap){
        case 'W': m_write_alloc_policy = WRITE_ALLOCATE; break;
        case 'N': m_write_alloc_policy = NO_WRITE_ALLOCATE; break;
        default: exit_parse_error();
        }

        // detect invalid configuration 
        if (m_alloc_policy == ON_FILL and m_write_policy == WRITE_BACK) {
            // A writeback cache with allocate-on-fill policy will inevitably lead to deadlock:  
            // The deadlock happens when an incoming cache-fill evicts a dirty
            // line, generating a writeback request.  If the memory subsystem
            // is congested, the interconnection network may not have
            // sufficient buffer for the writeback request.  This stalls the
            // incoming cache-fill.  The stall may propagate through the memory
            // subsystem back to the output port of the same core, creating a
            // deadlock where the wrtieback request and the incoming cache-fill
            // are stalling each other.  
            assert(0 && "Invalid cache configuration: Writeback cache cannot allocate new line on fill. "); 
        }

        // lld: set coalescing granularity
        if(m_mshr_type == LARGE_MSHR)
            m_coalesce_line_sz = 128;
        else
            m_coalesce_line_sz = m_line_sz;
        assert(m_line_sz % CACHE_CHUNK_SIZE == 0);
        assert(m_coalesce_line_sz % m_line_sz == 0);
        m_extra_accessq_size = 16;

        // default: port to data array width and granularity = line size 
        if (m_data_port_width == 0) {
            //m_data_port_width = m_line_sz; 
            m_data_port_width = m_coalesce_line_sz; // lld
        }
        //assert(m_line_sz % m_data_port_width == 0); // lld

        switch(sif){
        case 'H': m_set_index_function = FERMI_HASH_SET_FUNCTION; break;
        case 'C': m_set_index_function = CUSTOM_SET_FUNCTION; break;
        case 'L': m_set_index_function = LINEAR_SET_FUNCTION; break;
        default: exit_parse_error();
        }
    }
    bool disabled() const { return m_disabled;}
    unsigned get_line_sz() const
    {
        assert( m_valid );
        return m_line_sz;
    }
    unsigned get_num_lines() const
    {
        assert( m_valid );
        return m_nset * m_assoc;
    }

    void print( FILE *fp ) const
    {
        fprintf( fp, "Size = %d B (%d Set x %d-way x %d byte line)\n", 
                 m_line_sz * m_nset * m_assoc,
                 m_nset, m_assoc, m_line_sz );
    }

    virtual unsigned set_index( new_addr_type addr ) const
    {
        if(m_set_index_function != LINEAR_SET_FUNCTION){
            printf("\nGPGPU-Sim cache configuration error: Hashing or "
                    "custom set index function selected in configuration "
                    "file for a cache that has not overloaded the set_index "
                    "function\n");
            abort();
        }
        return(addr >> m_line_sz_log2) & (m_nset-1);
    }

    new_addr_type tag( new_addr_type addr ) const
    {
        // For generality, the tag includes both index and tag. This allows for more complex set index
        // calculations that can result in different indexes mapping to the same set, thus the full
        // tag + index is required to check for hit/miss. Tag is now identical to the block address.

        //return addr >> (m_line_sz_log2+m_nset_log2);
        return addr & ~(m_line_sz-1);
    }
    new_addr_type block_addr( new_addr_type addr ) const
    {
        return addr & ~(m_line_sz-1);
    }
    FuncCache get_cache_status() {return cache_status;}

    // lld: adaptive coalescing cache
    replacement_policy_t get_replacement_policy() { return m_replacement_policy; }
    unsigned get_coalesce_line_sz() const { assert( m_valid ); return m_coalesce_line_sz; }
    new_addr_type coalesced_block_addr( new_addr_type addr ) const
    {
        return addr & ~(m_coalesce_line_sz-1);
    }

    char *m_config_string;
    char *m_config_repl_string; // lld: replacement configuration
    char *m_config_stringPrefL1;
    char *m_config_stringPrefShared;
    FuncCache cache_status;

protected:
    void exit_parse_error()
    {
        printf("GPGPU-Sim uArch: cache configuration parsing error (%s)\n", m_config_string );
        abort();
    }

    bool m_valid;
    bool m_disabled;
    unsigned m_line_sz;
    unsigned m_line_sz_log2;
    unsigned m_nset;
    unsigned m_nset_log2;
    unsigned m_assoc;
    unsigned m_true_assoc; // lld: adaptive granularity
    unsigned m_coalesce_line_sz; // lld: memory coalescing granularity

    enum replacement_policy_t m_replacement_policy; // 'L' = LRU, 'F' = FIFO
                                                    // 'M' = MY_REPL, 'S' = SECTOR, 'G' = ADAP_GRAN, 'P' = PREF, 'T' = PARTITION
    unsigned m_custom_repl; // lld: 0 = lru, 1 = custom, 2 = other
    enum write_policy_t m_write_policy;             // 'T' = write through, 'B' = write back, 'R' = read only
    enum allocation_policy_t m_alloc_policy;        // 'm' = allocate on miss, 'f' = allocate on fill
    enum mshr_config_t m_mshr_type;

    write_allocate_policy_t m_write_alloc_policy;	// 'W' = Write allocate, 'N' = No write allocate

    union {
        unsigned m_mshr_entries;
        unsigned m_fragment_fifo_entries;
    };
    union {
        unsigned m_mshr_max_merge;
        unsigned m_request_fifo_entries;
    };
    union {
        unsigned m_miss_queue_size;
        unsigned m_rob_entries;
    };
    bool m_enable_prefetch;
    unsigned m_extra_accessq_size; // lld: extra access queue size
    unsigned m_result_fifo_entries;
    unsigned m_data_port_width; //< number of byte the cache can access per cycle 
    enum set_index_function m_set_index_function; // Hash, linear, or custom set index function

    friend class tag_array;
    friend class baseline_cache;
    friend class read_only_cache;
    friend class tex_cache;
    friend class data_cache;
    friend class l1_cache;
    friend class l2_cache;
    friend class BypassMonitor; // lld
};

class l1d_cache_config : public cache_config{
public:
	l1d_cache_config() : cache_config(){}
	virtual unsigned set_index(new_addr_type addr) const;
};

class l2_cache_config : public cache_config {
public:
	l2_cache_config() : cache_config(){}
	void init(linear_to_raw_address_translation *address_mapping);
	virtual unsigned set_index(new_addr_type addr) const;

private:
	linear_to_raw_address_translation *m_address_mapping;
};

// lld: variables and functions for replacement
#include "repl_policy.h"

class mshr_table {
public:
    mshr_table( unsigned num_entries, unsigned max_merged )
    : m_num_entries(num_entries),
    m_max_merged(max_merged)
#if (tr1_hash_map_ismap == 0)
    ,m_data(2*num_entries)
#endif
    {
        m_line_sz = 0;
        m_size_full = 0;
        m_list_full = 0;
    }

    /// Checks if there is a pending request to the lower memory level already
    bool probe( new_addr_type block_addr, mem_fetch *mf ) const; // lld: sector cache
    // lld: check if this request has been returned
    bool is_ready( new_addr_type block_addr, mem_fetch *mf ) const;
    /// Checks if there is space for tracking a new memory access
    bool full( new_addr_type block_addr );
    /// Add or merge this access
    void add( new_addr_type block_addr, mem_fetch *mf, std::bitset<MAX_CACHE_CHUNK_NUM> ready_sector_mask ); // lld: sector cache
    /// Returns true if cannot accept new fill responses
    bool busy() const {return false;}
    /// Accept a new cache fill response: mark entry ready for processing
    void mark_ready( mem_fetch *mf, new_addr_type block_addr, bool &has_atomic ); // lld: sector cache
    /// Returns true if ready accesses exist
    //bool access_ready() const {return !m_current_response.empty();}
    bool access_ready() const {if(m_type == SECTOR_MSHR || m_type == LARGE_MSHR) return !m_current_ready_request.empty();
                               else return !m_current_response.empty();} // lld: sector cache
    /// Returns next ready access
    mem_fetch *next_access();
    mem_fetch *sector_next_access(); // lld: sector cache
    void display( FILE *fp ) const;

    void check_mshr_parameters( unsigned num_entries, unsigned max_merged )
    {
    	assert(m_num_entries==num_entries && "Change of MSHR parameters between kernels is not allowed");
    	assert(m_max_merged==max_merged && "Change of MSHR parameters between kernels is not allowed");
    }

    // lld: adaptive granularity
    void set_type(enum mshr_config_t type) { m_type = type; }
    void set_line_sz(unsigned line_sz) { m_line_sz = line_sz; }
    void set_cache_line_sz(unsigned line_sz) { m_cache_line_sz = line_sz; }
    new_addr_type get_block_addr( new_addr_type addr ) const
    {
        return addr & ~(m_line_sz-1);
    }
    new_addr_type get_cache_block_addr( new_addr_type addr ) const
    {
        return addr & ~(m_cache_line_sz-1);
    }
    void set_l2(bool l2) { m_l2=l2; }
    void get_sub_stats(struct cache_sub_stats &css) const;

private:

    // finite sized, fully associative table, with a finite maximum number of merged requests
    const unsigned m_num_entries;
    const unsigned m_max_merged;

    struct mshr_entry {
        std::list<mem_fetch*> m_list;
        bool m_has_atomic; 
        std::bitset<MAX_CACHE_CHUNK_NUM> m_sector_mask; // lld: sector cache
        std::bitset<MAX_CACHE_CHUNK_NUM> m_ready_sector_mask; // lld: sector cache
        std::bitset<MAX_CACHE_CHUNK_NUM> m_original_ready_sector_mask; // lld: sector cache
        mshr_entry() : m_has_atomic(false) { }
    }; 
    typedef tr1_hash_map<new_addr_type,mshr_entry> table;
    table m_data;

    // it may take several cycles to process the merged requests
    bool m_current_response_ready;
    std::list<new_addr_type> m_current_response;
    std::list<mem_fetch*> m_current_ready_request; // lld: sector cache
    std::list<mem_fetch*> m_other_ready_request; // lld: sector cache
    // lld: adaptive granularity
    enum mshr_config_t m_type;
    unsigned m_line_sz;
    unsigned m_cache_line_sz;
    bool m_l2;

    unsigned m_size_full;
    unsigned m_list_full;
};


/***************************************************************** Caches *****************************************************************/
///
/// Simple struct to maintain cache accesses, misses, pending hits, and reservation fails.
///
struct cache_sub_stats{
    unsigned accesses;
    unsigned misses;
    unsigned pending_hits;
    unsigned res_fails;

    // lld
    unsigned partial_misses;
    unsigned fills;
    unsigned long long total_lat;
    unsigned hit_fills;
    unsigned long long total_hit_lat;
    unsigned mshr_size_full;
    unsigned mshr_list_full;
    unsigned prefetches;
    unsigned useful_prefetches;
    unsigned useless_prefetches;
    unsigned desired_prefetches;
    unsigned miss_repl;
    unsigned partial_miss_repl;
    unsigned reuse[MAX_MEMORY_ACCESS_SIZE+1];
    unsigned chunk_reuse[MAX_CACHE_CHUNK_NUM+1];
    unsigned chunk_cont_reuse[MAX_CACHE_CHUNK_NUM+1];
    unsigned repl_stats[NUM_CACHE_REPL_STATS];

    unsigned long long port_available_cycles; 
    unsigned long long data_port_busy_cycles; 
    unsigned long long fill_port_busy_cycles; 

    cache_sub_stats(){
        clear();
    }
    void clear(){
        accesses = 0;
        misses = 0;
        pending_hits = 0;
        res_fails = 0;
        port_available_cycles = 0; 
        data_port_busy_cycles = 0; 
        fill_port_busy_cycles = 0; 
        // lld
        partial_misses = 0;
        fills = 0;
        total_lat = 0;
        hit_fills = 0;
        total_hit_lat = 0;
        mshr_size_full = 0;
        mshr_list_full = 0;
        prefetches = 0;
        useful_prefetches = 0;
        useless_prefetches = 0;
        desired_prefetches = 0;
        miss_repl = 0;
        partial_miss_repl = 0;
        for(unsigned i = 0; i <= MAX_MEMORY_ACCESS_SIZE; i++)
            reuse[i] = 0;
        for(unsigned i = 0; i <= MAX_CACHE_CHUNK_NUM; i++) {
            chunk_reuse[i] = 0;
            chunk_cont_reuse[i] = 0;
        }
        for(int i = 0; i < NUM_CACHE_REPL_STATS; i++)
            repl_stats[i] = 0;
    }
    cache_sub_stats &operator+=(const cache_sub_stats &css){
        ///
        /// Overloading += operator to easily accumulate stats
        ///
        accesses += css.accesses;
        misses += css.misses;
        pending_hits += css.pending_hits;
        res_fails += css.res_fails;
        port_available_cycles += css.port_available_cycles; 
        data_port_busy_cycles += css.data_port_busy_cycles; 
        fill_port_busy_cycles += css.fill_port_busy_cycles; 
        // lld
        partial_misses += css.partial_misses;
        fills += css.fills;
        total_lat += css.total_lat;
        hit_fills += css.hit_fills;
        total_hit_lat += css.total_hit_lat;
        mshr_size_full += css.mshr_size_full;
        mshr_list_full += css.mshr_list_full;
        prefetches += css.prefetches;
        useful_prefetches += css.useful_prefetches;
        useless_prefetches += css.useless_prefetches;
        desired_prefetches += css.desired_prefetches;
        miss_repl += css.miss_repl;
        partial_miss_repl += css.partial_miss_repl;
        for(unsigned i = 0; i <= MAX_MEMORY_ACCESS_SIZE; i++)
            reuse[i] += css.reuse[i];
        for(unsigned i = 0; i <= MAX_CACHE_CHUNK_NUM; i++) {
            chunk_reuse[i] += css.chunk_reuse[i];
            chunk_cont_reuse[i] += css.chunk_cont_reuse[i];
        }
        for(int i = 0; i < NUM_CACHE_REPL_STATS; i++)
            repl_stats[i] += css.repl_stats[i];
        return *this;
    }

    cache_sub_stats operator+(const cache_sub_stats &cs){
        ///
        /// Overloading + operator to easily accumulate stats
        ///
        cache_sub_stats ret;
        ret.accesses = accesses + cs.accesses;
        ret.misses = misses + cs.misses;
        ret.pending_hits = pending_hits + cs.pending_hits;
        ret.res_fails = res_fails + cs.res_fails;
        ret.port_available_cycles = port_available_cycles + cs.port_available_cycles; 
        ret.data_port_busy_cycles = data_port_busy_cycles + cs.data_port_busy_cycles; 
        ret.fill_port_busy_cycles = fill_port_busy_cycles + cs.fill_port_busy_cycles; 
        // lld
        ret.partial_misses = partial_misses + cs.partial_misses;
        ret.fills = fills + cs.fills;
        ret.total_lat = total_lat + cs.total_lat;
        ret.hit_fills = hit_fills + cs.hit_fills;
        ret.total_hit_lat = total_hit_lat + cs.total_hit_lat;
        ret.mshr_size_full = mshr_size_full + cs.mshr_size_full;
        ret.mshr_list_full = mshr_list_full + cs.mshr_list_full;
        ret.prefetches = prefetches + cs.prefetches;
        ret.useful_prefetches = useful_prefetches + cs.useful_prefetches;
        ret.useless_prefetches = useless_prefetches + cs.useless_prefetches;
        ret.desired_prefetches = desired_prefetches + cs.desired_prefetches;
        ret.miss_repl = miss_repl + cs.miss_repl;
        ret.partial_miss_repl = partial_miss_repl + cs.partial_miss_repl;
        for(unsigned i = 0; i <= MAX_MEMORY_ACCESS_SIZE; i++)
            ret.reuse[i] = reuse[i] + cs.reuse[i];
        for(unsigned i = 0; i <= MAX_CACHE_CHUNK_NUM; i++) {
            ret.chunk_reuse[i] = chunk_reuse[i] + cs.chunk_reuse[i];
            ret.chunk_cont_reuse[i] = chunk_cont_reuse[i] + cs.chunk_cont_reuse[i];
        }
        for(int i = 0; i < NUM_CACHE_REPL_STATS; i++)
            ret.repl_stats[i] = repl_stats[i] + cs.repl_stats[i];
        return ret;
    }

    void print_port_stats(FILE *fout, const char *cache_name) const; 
};

///
/// Cache_stats
/// Used to record statistics for each cache.
/// Maintains a record of every 'mem_access_type' and its resulting
/// 'cache_request_status' : [mem_access_type][cache_request_status]
///
class cache_stats {
public:
    cache_stats();
    void clear();
    void inc_stats(int access_type, int access_outcome);
    enum cache_request_status select_stats_status(enum cache_request_status probe, enum cache_request_status access) const;
    unsigned &operator()(int access_type, int access_outcome);
    unsigned operator()(int access_type, int access_outcome) const;
    cache_stats operator+(const cache_stats &cs);
    cache_stats &operator+=(const cache_stats &cs);
    void print_stats(FILE *fout, const char *cache_name = "Cache_stats") const;

    unsigned get_stats(enum mem_access_type *access_type, unsigned num_access_type, enum cache_request_status *access_status, unsigned num_access_status)  const;
    void get_sub_stats(struct cache_sub_stats &css) const;

    void sample_cache_port_utility(bool data_port_busy, bool fill_port_busy); 
private:
    bool check_valid(int type, int status) const;

    std::vector< std::vector<unsigned> > m_stats;

    unsigned long long m_cache_port_available_cycles; 
    unsigned long long m_cache_data_port_busy_cycles; 
    unsigned long long m_cache_fill_port_busy_cycles; 
};

class cache_t {
public:
    virtual ~cache_t() {}
    virtual enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ) =  0;
    virtual void spare_cycle( unsigned time ) =  0; // lld: spare cycles

    // accessors for cache bandwidth availability 
    virtual bool data_port_free() const = 0; 
    virtual bool fill_port_free() const = 0; 
};

bool was_write_sent( const std::list<cache_event> &events );
bool was_read_sent( const std::list<cache_event> &events );

/// Baseline cache
/// Implements common functions for read_only_cache and data_cache
/// Each subclass implements its own 'access' function
class baseline_cache : public cache_t {
public:
    baseline_cache( const char *name, cache_config &config, int core_id, int type_id, mem_fetch_interface *memport,
                     enum mem_fetch_status status )
    : m_config(config), m_tag_array(new tag_array(config,core_id,type_id)), 
      m_mshrs(config.m_mshr_entries,config.m_mshr_max_merge), 
      m_bandwidth_management(config) 
    {
        init( name, config, memport, status );
    }

    void init( const char *name,
               const cache_config &config,
               mem_fetch_interface *memport,
               enum mem_fetch_status status )
    {
        m_name = name;
        assert(config.m_mshr_type >= ASSOC); // lld
        m_memport=memport;
        m_miss_queue_status = status;
        m_mshrs.set_type(config.m_mshr_type);
        m_mshrs.set_line_sz(config.get_coalesce_line_sz());
        m_mshrs.set_cache_line_sz(config.get_line_sz());
    }

    virtual ~baseline_cache()
    {
        delete m_tag_array;
    }

	void update_cache_parameters(cache_config &config)
	{
		m_config=config;
		m_tag_array->update_cache_parameters(config);
		m_mshrs.check_mshr_parameters(config.m_mshr_entries,config.m_mshr_max_merge);
	}

    virtual enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ) =  0;
    virtual void spare_cycle( unsigned time ) =  0; // lld: spare cycles
    /// Sends next request to lower level of memory
    void cycle();
    /// Interface for response from lower memory level (model bandwidth restictions in caller)
    void fill( mem_fetch *mf, unsigned time );
    /// Checks if mf is waiting to be filled by lower memory level
    bool waiting_for_fill( mem_fetch *mf );
    /// Are any (accepted) accesses that had to wait for memory now ready? (does not include accesses that "HIT")
    bool access_ready() const {return m_mshrs.access_ready();}
    /// Pop next ready access (does not include accesses that "HIT")
    mem_fetch *next_access(){return m_mshrs.next_access();}
    // flash invalidate all entries in cache
    void flush(){m_tag_array->flush();}
    void print(FILE *fp, unsigned &accesses, unsigned &misses) const;
    void print_stats() const; // lld: print replacement stats
    void display_state( FILE *fp ) const;

    // Stat collection
    const cache_stats &get_stats() const {
        return m_stats;
    }
    unsigned get_stats(enum mem_access_type *access_type, unsigned num_access_type, enum cache_request_status *access_status, unsigned num_access_status)  const{
        return m_stats.get_stats(access_type, num_access_type, access_status, num_access_status);
    }
    void get_sub_stats(struct cache_sub_stats &css) const {
        m_stats.get_sub_stats(css);
        m_mshrs.get_sub_stats(css);
        m_tag_array->get_sub_stats(css); // lld: get replacement stats
    }

    // accessors for cache bandwidth availability 
    bool data_port_free() const { return m_bandwidth_management.data_port_free(); } 
    bool fill_port_free() const { return m_bandwidth_management.fill_port_free(); } 

protected:
    // Constructor that can be used by derived classes with custom tag arrays
    baseline_cache( const char *name,
                    cache_config &config,
                    int core_id,
                    int type_id,
                    mem_fetch_interface *memport,
                    enum mem_fetch_status status,
                    tag_array* new_tag_array )
    : m_config(config),
      m_tag_array( new_tag_array ),
      m_mshrs(config.m_mshr_entries,config.m_mshr_max_merge), 
      m_bandwidth_management(config) 
    {
        init( name, config, memport, status );
    }

protected:
    std::string m_name;
    cache_config &m_config;
    tag_array*  m_tag_array;
    mshr_table m_mshrs;
    std::list<mem_fetch*> m_miss_queue;
    enum mem_fetch_status m_miss_queue_status;
    mem_fetch_interface *m_memport;

    struct extra_mf_fields {
        extra_mf_fields()  { m_valid = false;}
        extra_mf_fields( new_addr_type a, unsigned i, unsigned d ) 
        {
            m_valid = true;
            m_block_addr = a;
            m_cache_index = i;
            m_data_size = d;
        }
        bool m_valid;
        new_addr_type m_block_addr;
        unsigned m_cache_index;
        unsigned m_data_size;
    };

    typedef std::map<mem_fetch*,extra_mf_fields> extra_mf_fields_lookup;

    extra_mf_fields_lookup m_extra_mf_fields;

    std::list<mem_fetch*> m_extra_accessq; // lld: extra access queue
    std::list<mem_fetch*> m_extra_pending_list; // lld: list to record pending extra requests

    cache_stats m_stats;

    /// Checks whether this request can be handled on this cycle. num_miss equals max # of misses to be handled on this cycle
    bool miss_queue_full(unsigned num_miss){
    	  return ( (m_miss_queue.size()+num_miss) >= m_config.m_miss_queue_size );
    }
    /// Read miss handler without writeback
    void send_read_request(new_addr_type addr, new_addr_type block_addr, unsigned cache_index, mem_fetch *mf,
    		unsigned time, bool &do_miss, std::list<cache_event> &events, bool read_only, bool wa);
    /// Read miss handler. Check MSHR hit or MSHR available
    void send_read_request(new_addr_type addr, new_addr_type block_addr, unsigned cache_index, mem_fetch *mf,
    		unsigned time, bool &do_miss, bool &wb, cache_block_t &evicted, std::list<cache_event> &events, bool read_only, bool wa);
    void issue_request(new_addr_type block_addr, unsigned cache_index, mem_fetch *mf, unsigned time, std::list<cache_event> &events, bool wa); // lld: try to combine requests

    /// Sub-class containing all metadata for port bandwidth management 
    class bandwidth_management 
    {
    public: 
        bandwidth_management(cache_config &config); 

        /// use the data port based on the outcome and events generated by the mem_fetch request 
        void use_data_port(mem_fetch *mf, enum cache_request_status outcome, const std::list<cache_event> &events); 

        /// use the fill port 
        void use_fill_port(mem_fetch *mf); 

        /// called every cache cycle to free up the ports 
        void replenish_port_bandwidth(); 

        /// query for data port availability 
        bool data_port_free() const; 
        /// query for fill port availability 
        bool fill_port_free() const; 
    protected: 
        const cache_config &m_config; 

        int m_data_port_occupied_cycles; //< Number of cycle that the data port remains used 
        int m_fill_port_occupied_cycles; //< Number of cycle that the fill port remains used 
    }; 

    bandwidth_management m_bandwidth_management; 
    bool m_l2; // lld: whether this is l2 cache
};

/// Read only cache
class read_only_cache : public baseline_cache {
public:
    read_only_cache( const char *name, cache_config &config, int core_id, int type_id, mem_fetch_interface *memport, enum mem_fetch_status status )
    : baseline_cache(name,config,core_id,type_id,memport,status){}

    /// Access cache for read_only_cache: returns RESERVATION_FAIL if request could not be accepted (for any reason)
    virtual enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events );
    void spare_cycle( unsigned time ) {} // lld: spare cycles

    virtual ~read_only_cache(){}

protected:
    read_only_cache( const char *name, cache_config &config, int core_id, int type_id, mem_fetch_interface *memport, enum mem_fetch_status status, tag_array* new_tag_array )
    : baseline_cache(name,config,core_id,type_id,memport,status, new_tag_array){}
};

/// Data cache - Implements common functions for L1 and L2 data cache
class data_cache : public baseline_cache {
public:
    data_cache( const char *name, cache_config &config,
    			int core_id, int type_id, mem_fetch_interface *memport,
                mem_fetch_allocator *mfcreator, enum mem_fetch_status status,
                mem_access_type wr_alloc_type, mem_access_type wrbk_type )
    			: baseline_cache(name,config,core_id,type_id,memport,status)
    {
        init( mfcreator );
        m_wr_alloc_type = wr_alloc_type;
        m_wrbk_type = wrbk_type;
    }

    virtual ~data_cache() {}

    virtual void init( mem_fetch_allocator *mfcreator )
    {
        m_memfetch_creator=mfcreator;

        // Set read hit function
        m_rd_hit = &data_cache::rd_hit_base;

        // Set read miss function
        m_rd_miss = &data_cache::rd_miss_base;

        // Set write hit function
        switch(m_config.m_write_policy){
        // READ_ONLY is now a separate cache class, config is deprecated
        case READ_ONLY:
            assert(0 && "Error: Writable Data_cache set as READ_ONLY\n");
            break; 
        case WRITE_BACK: m_wr_hit = &data_cache::wr_hit_wb; break;
        case WRITE_THROUGH: m_wr_hit = &data_cache::wr_hit_wt; break;
        case WRITE_EVICT: m_wr_hit = &data_cache::wr_hit_we; break;
        case LOCAL_WB_GLOBAL_WT:
            m_wr_hit = &data_cache::wr_hit_global_we_local_wb;
            break;
        default:
            assert(0 && "Error: Must set valid cache write policy\n");
            break; // Need to set a write hit function
        }

        // Set write miss function
        switch(m_config.m_write_alloc_policy){
        case WRITE_ALLOCATE: m_wr_miss = &data_cache::wr_miss_wa; break;
        case NO_WRITE_ALLOCATE: m_wr_miss = &data_cache::wr_miss_no_wa; break;
        default:
            assert(0 && "Error: Must set valid cache write miss policy\n");
            break; // Need to set a write miss function
        }
    }

    virtual enum cache_request_status access( new_addr_type addr,
                                              mem_fetch *mf,
                                              unsigned time,
                                              std::list<cache_event> &events );
    virtual void spare_cycle( unsigned time ); // lld: spare cycles
protected:
    data_cache( const char *name,
                cache_config &config,
                int core_id,
                int type_id,
                mem_fetch_interface *memport,
                mem_fetch_allocator *mfcreator,
                enum mem_fetch_status status,
                tag_array* new_tag_array,
                mem_access_type wr_alloc_type,
                mem_access_type wrbk_type)
    : baseline_cache(name, config, core_id, type_id, memport,status, new_tag_array)
    {
        init( mfcreator );
        m_wr_alloc_type = wr_alloc_type;
        m_wrbk_type = wrbk_type;
    }

    mem_access_type m_wr_alloc_type; // Specifies type of write allocate request (e.g., L1 or L2)
    mem_access_type m_wrbk_type; // Specifies type of writeback request (e.g., L1 or L2)

    //! A general function that takes the result of a tag_array probe
    //  and performs the correspding functions based on the cache configuration
    //  The access fucntion calls this function
    enum cache_request_status
        process_tag_probe( bool wr,
                           enum cache_request_status status,
                           new_addr_type addr,
                           unsigned cache_index,
                           mem_fetch* mf,
                           unsigned time,
                           std::list<cache_event>& events );

protected:
    mem_fetch_allocator *m_memfetch_creator;

    // Functions for data cache access
    /// Sends write request to lower level memory (write or writeback)
    void send_write_request( mem_fetch *mf,
                             cache_event request,
                             unsigned time,
                             std::list<cache_event> &events);

    // Member Function pointers - Set by configuration options
    // to the functions below each grouping
    /******* Write-hit configs *******/
    enum cache_request_status
        (data_cache::*m_wr_hit)( new_addr_type addr,
                                 unsigned cache_index,
                                 mem_fetch *mf,
                                 unsigned time,
                                 std::list<cache_event> &events,
                                 enum cache_request_status status );
    /// Marks block as MODIFIED and updates block LRU
    enum cache_request_status
        wr_hit_wb( new_addr_type addr,
                   unsigned cache_index,
                   mem_fetch *mf,
                   unsigned time,
                   std::list<cache_event> &events,
                   enum cache_request_status status ); // write-back
    enum cache_request_status
        wr_hit_wt( new_addr_type addr,
                   unsigned cache_index,
                   mem_fetch *mf,
                   unsigned time,
                   std::list<cache_event> &events,
                   enum cache_request_status status ); // write-through

    /// Marks block as INVALID and sends write request to lower level memory
    enum cache_request_status
        wr_hit_we( new_addr_type addr,
                   unsigned cache_index,
                   mem_fetch *mf,
                   unsigned time,
                   std::list<cache_event> &events,
                   enum cache_request_status status ); // write-evict
    enum cache_request_status
        wr_hit_global_we_local_wb( new_addr_type addr,
                                   unsigned cache_index,
                                   mem_fetch *mf,
                                   unsigned time,
                                   std::list<cache_event> &events,
                                   enum cache_request_status status );
        // global write-evict, local write-back


    /******* Write-miss configs *******/
    enum cache_request_status
        (data_cache::*m_wr_miss)( new_addr_type addr,
                                  unsigned cache_index,
                                  mem_fetch *mf,
                                  unsigned time,
                                  std::list<cache_event> &events,
                                  enum cache_request_status status );
    /// Sends read request, and possible write-back request,
    //  to lower level memory for a write miss with write-allocate
    enum cache_request_status
        wr_miss_wa( new_addr_type addr,
                    unsigned cache_index,
                    mem_fetch *mf,
                    unsigned time,
                    std::list<cache_event> &events,
                    enum cache_request_status status ); // write-allocate
    enum cache_request_status
        wr_miss_no_wa( new_addr_type addr,
                       unsigned cache_index,
                       mem_fetch *mf,
                       unsigned time,
                       std::list<cache_event> &events,
                       enum cache_request_status status ); // no write-allocate

    // Currently no separate functions for reads
    /******* Read-hit configs *******/
    enum cache_request_status
        (data_cache::*m_rd_hit)( new_addr_type addr,
                                 unsigned cache_index,
                                 mem_fetch *mf,
                                 unsigned time,
                                 std::list<cache_event> &events,
                                 enum cache_request_status status );
    enum cache_request_status
        rd_hit_base( new_addr_type addr,
                     unsigned cache_index,
                     mem_fetch *mf,
                     unsigned time,
                     std::list<cache_event> &events,
                     enum cache_request_status status );

    /******* Read-miss configs *******/
    enum cache_request_status
        (data_cache::*m_rd_miss)( new_addr_type addr,
                                  unsigned cache_index,
                                  mem_fetch *mf,
                                  unsigned time,
                                  std::list<cache_event> &events,
                                  enum cache_request_status status );
    enum cache_request_status
        rd_miss_base( new_addr_type addr,
                      unsigned cache_index,
                      mem_fetch*mf,
                      unsigned time,
                      std::list<cache_event> &events,
                      enum cache_request_status status );

    void prefetch( mem_fetch *mf, unsigned time, enum cache_request_status probe_status ); // lld: generate prefetch requests

};

/// This is meant to model the first level data cache in Fermi.
/// It is write-evict (global) or write-back (local) at
/// the granularity of individual blocks
/// (the policy used in fermi according to the CUDA manual)
class l1_cache : public data_cache {
public:
    l1_cache(const char *name, cache_config &config,
            int core_id, int type_id, mem_fetch_interface *memport,
            mem_fetch_allocator *mfcreator, enum mem_fetch_status status )
            : data_cache(name,config,core_id,type_id,memport,mfcreator,status, L1_WR_ALLOC_R, L1_WRBK_ACC){ m_l2=false; m_mshrs.set_l2(m_l2); }

    virtual ~l1_cache(){}

    virtual enum cache_request_status
        access( new_addr_type addr,
                mem_fetch *mf,
                unsigned time,
                std::list<cache_event> &events );
    virtual void spare_cycle( unsigned time ); // lld: spare cycles

protected:
    l1_cache( const char *name,
              cache_config &config,
              int core_id,
              int type_id,
              mem_fetch_interface *memport,
              mem_fetch_allocator *mfcreator,
              enum mem_fetch_status status,
              tag_array* new_tag_array )
    : data_cache( name,
                  config,
                  core_id,type_id,memport,mfcreator,status, new_tag_array, L1_WR_ALLOC_R, L1_WRBK_ACC ){ m_l2=false; m_mshrs.set_l2(m_l2); }

};

/// Models second level shared cache with global write-back
/// and write-allocate policies
class l2_cache : public data_cache {
public:
    l2_cache(const char *name,  cache_config &config,
            int core_id, int type_id, mem_fetch_interface *memport,
            mem_fetch_allocator *mfcreator, enum mem_fetch_status status )
            : data_cache(name,config,core_id,type_id,memport,mfcreator,status, L2_WR_ALLOC_R, L2_WRBK_ACC){ m_l2=true; m_mshrs.set_l2(m_l2); }

    virtual ~l2_cache() {}

    virtual enum cache_request_status
        access( new_addr_type addr,
                mem_fetch *mf,
                unsigned time,
                std::list<cache_event> &events );
    virtual void spare_cycle( unsigned time ); // lld: spare cycles
};

/*****************************************************************************/

// See the following paper to understand this cache model:
// 
// Igehy, et al., Prefetching in a Texture Cache Architecture, 
// Proceedings of the 1998 Eurographics/SIGGRAPH Workshop on Graphics Hardware
// http://www-graphics.stanford.edu/papers/texture_prefetch/
class tex_cache : public cache_t {
public:
    tex_cache( const char *name, cache_config &config, int core_id, int type_id, mem_fetch_interface *memport,
               enum mem_fetch_status request_status, 
               enum mem_fetch_status rob_status )
    : m_config(config), 
    m_tags(config,core_id,type_id), 
    m_fragment_fifo(config.m_fragment_fifo_entries), 
    m_request_fifo(config.m_request_fifo_entries),
    m_rob(config.m_rob_entries),
    m_result_fifo(config.m_result_fifo_entries)
    {
        m_name = name;
        assert(config.m_mshr_type == TEX_FIFO);
        assert(config.m_write_policy == READ_ONLY);
        assert(config.m_alloc_policy == ON_MISS);
        m_memport=memport;
        m_cache = new data_block[ config.get_num_lines() ];
        m_request_queue_status = request_status;
        m_rob_status = rob_status;
    }

    /// Access function for tex_cache
    /// return values: RESERVATION_FAIL if request could not be accepted
    /// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
    /// since unlike a normal CPU cache, a "HIT" in texture cache does not
    /// mean the data is ready (still need to get through fragment fifo)
    enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events );
    void spare_cycle( unsigned time ) {} // lld: spare cycles
    void cycle();
    /// Place returning cache block into reorder buffer
    void fill( mem_fetch *mf, unsigned time );
    /// Are any (accepted) accesses that had to wait for memory now ready? (does not include accesses that "HIT")
    bool access_ready() const{return !m_result_fifo.empty();}
    /// Pop next ready access (includes both accesses that "HIT" and those that "MISS")
    mem_fetch *next_access(){return m_result_fifo.pop();}
    void display_state( FILE *fp ) const;

    // accessors for cache bandwidth availability - stubs for now 
    bool data_port_free() const { return true; }
    bool fill_port_free() const { return true; }

    // Stat collection
    const cache_stats &get_stats() const {
        return m_stats;
    }
    unsigned get_stats(enum mem_access_type *access_type, unsigned num_access_type, enum cache_request_status *access_status, unsigned num_access_status) const{
        return m_stats.get_stats(access_type, num_access_type, access_status, num_access_status);
    }

    void get_sub_stats(struct cache_sub_stats &css) const{
        m_stats.get_sub_stats(css);
    }
private:
    std::string m_name;
    const cache_config &m_config;

    struct fragment_entry {
        fragment_entry() {}
        fragment_entry( mem_fetch *mf, unsigned idx, bool m, unsigned d )
        {
            m_request=mf;
            m_cache_index=idx;
            m_miss=m;
            m_data_size=d;
        }
        mem_fetch *m_request;     // request information
        unsigned   m_cache_index; // where to look for data
        bool       m_miss;        // true if sent memory request
        unsigned   m_data_size;
    };

    struct rob_entry {
        rob_entry() { m_ready = false; m_time=0; m_request=NULL;}
        rob_entry( unsigned i, mem_fetch *mf, new_addr_type a ) 
        { 
            m_ready=false; 
            m_index=i;
            m_time=0;
            m_request=mf; 
            m_block_addr=a;
        }
        bool m_ready;
        unsigned m_time; // which cycle did this entry become ready?
        unsigned m_index; // where in cache should block be placed?
        mem_fetch *m_request;
        new_addr_type m_block_addr;
    };

    struct data_block {
        data_block() { m_valid = false;}
        bool m_valid;
        new_addr_type m_block_addr;
    };

    // TODO: replace fifo_pipeline with this?
    template<class T> class fifo {
    public:
        fifo( unsigned size ) 
        { 
            m_size=size; 
            m_num=0; 
            m_head=0; 
            m_tail=0; 
            m_data = new T[size];
        }
        bool full() const { return m_num == m_size;}
        bool empty() const { return m_num == 0;}
        unsigned size() const { return m_num;}
        unsigned capacity() const { return m_size;}
        unsigned push( const T &e ) 
        { 
            assert(!full()); 
            m_data[m_head] = e; 
            unsigned result = m_head;
            inc_head(); 
            return result;
        }
        T pop() 
        { 
            assert(!empty()); 
            T result = m_data[m_tail];
            inc_tail();
            return result;
        }
        const T &peek( unsigned index ) const 
        { 
            assert( index < m_size );
            return m_data[index]; 
        }
        T &peek( unsigned index ) 
        { 
            assert( index < m_size );
            return m_data[index]; 
        }
        T &peek() const
        { 
            return m_data[m_tail]; 
        }
        unsigned next_pop_index() const 
        {
            return m_tail;
        }
    private:
        void inc_head() { m_head = (m_head+1)%m_size; m_num++;}
        void inc_tail() { assert(m_num>0); m_tail = (m_tail+1)%m_size; m_num--;}

        unsigned   m_head; // next entry goes here
        unsigned   m_tail; // oldest entry found here
        unsigned   m_num;  // how many in fifo?
        unsigned   m_size; // maximum number of entries in fifo
        T         *m_data;
    };

    tag_array               m_tags;
    fifo<fragment_entry>    m_fragment_fifo;
    fifo<mem_fetch*>        m_request_fifo;
    fifo<rob_entry>         m_rob;
    data_block             *m_cache;
    fifo<mem_fetch*>        m_result_fifo; // next completed texture fetch

    mem_fetch_interface    *m_memport;
    enum mem_fetch_status   m_request_queue_status;
    enum mem_fetch_status   m_rob_status;

    struct extra_mf_fields {
        extra_mf_fields()  { m_valid = false;}
        extra_mf_fields( unsigned i ) 
        {
            m_valid = true;
            m_rob_index = i;
        }
        bool m_valid;
        unsigned m_rob_index;
    };

    cache_stats m_stats;

    typedef std::map<mem_fetch*,extra_mf_fields> extra_mf_fields_lookup;

    extra_mf_fields_lookup m_extra_mf_fields;
};

const char * cache_repl_stats_str(enum cache_repl_stats id); // lld

#endif
