enum cache_repl_type {
    BASE_REPL_LRU = 0,
    BASE_REPL_NRU,
    BASE_REPL_SRRIP,
    BASE_REPL_RANDOM,
    BASE_REPL_PDP
};

// Replacement State Per Cache Line
typedef struct
{
    unsigned LRUstackposition;
    bool NRUbit;
    unsigned RRPV;
    unsigned RPD; // remaining PD
    unsigned tid;
    unsigned used_num;
    unsigned cid;

} LINE_REPLACEMENT_STATE;

#define CACHE_REPL_STATS_TUP_DEF \
CRS_TUP_BEGIN( cache_repl_stats ) \
   CRS_TUP( INTRA_HIT ), \
   CRS_TUP( INTER_HIT ), \
   CRS_TUP( BYPASS ), \
   CRS_TUP( NONBYPASS ), \
   CRS_TUP( NUM_CACHE_REPL_STATS ) \
CRS_TUP_END( mem_access_type ) 

#define CRS_TUP_BEGIN(X) enum X {
#define CRS_TUP(X) X
#define CRS_TUP_END(X) };
CACHE_REPL_STATS_TUP_DEF
#undef CRS_TUP_BEGIN
#undef CRS_TUP
#undef CRS_TUP_END

class tag_array {
public:
    // Use this constructor
    tag_array(cache_config &config, int core_id, int type_id );
    ~tag_array();

    enum cache_request_status probe( mem_fetch *mf, new_addr_type addr, unsigned &idx );
    enum cache_request_status access( mem_fetch *mf, new_addr_type addr, unsigned time, unsigned &idx );
    enum cache_request_status access( mem_fetch *mf, new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted );

    unsigned fill( mem_fetch *mf, new_addr_type addr, unsigned time );
    unsigned fill( mem_fetch *mf, new_addr_type addr, unsigned idx, unsigned time );

    enum cache_request_status probe_selector( mem_fetch *mf, new_addr_type addr, unsigned &idx, std::list<unsigned> &repl_set, unsigned time );
    enum cache_request_status access_probe( mem_fetch *mf, new_addr_type addr, unsigned &idx, std::list<unsigned> &repl_set, unsigned time );
    enum cache_request_status fill_probe( mem_fetch *mf, new_addr_type addr, unsigned &idx );
    unsigned get_victim(mem_fetch *mf, new_addr_type addr);
    unsigned get_partial_victim(mem_fetch *mf, new_addr_type addr, unsigned idx); // partial_miss
    unsigned get_real_victim(mem_fetch *mf, new_addr_type addr, unsigned idx); // cannot bypass
    void update(mem_fetch *mf, new_addr_type addr, unsigned idx, enum cache_request_status status);
    void touch(mem_fetch *mf, new_addr_type addr); // touch address
    void decide_allocation(mem_fetch *mf, new_addr_type addr); // decide which parts should be allocated
    void decide_prefetch(mem_fetch *mf, new_addr_type addr); // decide which parts should be prefetched
    bool check_avail(mem_fetch *mf, new_addr_type addr); // check reservation fail

    unsigned size() const { return m_config.get_num_lines();}
    cache_block_t &get_block(unsigned idx) { return m_lines[idx];}

    void flush(); // flash invalidate all entries
    void new_window();

    void print( FILE *stream, unsigned &total_access, unsigned &total_misses ) const;
    void print_stats( std::ostream &out, std::string name ) const; // lld: print replacement stats
    float windowed_miss_rate( ) const;
    void get_stats(unsigned &total_access, unsigned &total_misses, unsigned &total_hit_res, unsigned &total_res_fail) const;
    cache_block_t* get_line(unsigned idx) { assert(idx != (unsigned)-1); return &m_lines[idx]; }

	void update_cache_parameters(cache_config &config);
    void get_sub_stats(struct cache_sub_stats &css) const; // lld: get replacement stats
protected:
    // This constructor is intended for use only from derived classes that wish to
    // avoid unnecessary memory allocation that takes place in the
    // other tag_array constructor
    tag_array( cache_config &config,
               int core_id,
               int type_id,
               cache_block_t* new_lines );
    void init( int core_id, int type_id );
    void init_repl(); // lld: initialize replacement status
    void update_stat(mem_fetch *mf, unsigned idx, bool hit); // lld: update statistics
    void update_sets(unsigned set_index); // lld: update set status

    // new functions
    int  Get_Random_Victim( unsigned setIndex, unsigned used_idx );
    int  Get_LRU_Victim( unsigned setIndex, unsigned used_idx, unsigned cid );
    void UpdateLRU( unsigned setIndex, int updateWayID );
    int  Get_NRU_Victim( unsigned setIndex, unsigned used_idx );
    void UpdateNRU( unsigned setIndex, int updateWayID );
    int  Get_RRIP_Victim( unsigned setIndex, unsigned used_idx );
    void UpdateSRRIP(unsigned setIndex, int updateWayID, bool cacheHit);

protected:

    cache_config &m_config;

    cache_block_t *m_lines; /* nbanks x nset x assoc lines in total */

    cache_set_t *m_sets; // lld: record set status

    unsigned m_access;
    unsigned m_miss;
    unsigned m_pending_hit; // number of cache miss that hit a line that is allocated but not filled
    unsigned m_res_fail;

    // lld
    unsigned m_fill;
    unsigned m_total_lat;
    unsigned m_hit_fill;
    unsigned m_total_hit_lat;
    unsigned m_intra_warp_hit;
    unsigned m_inter_warp_hit;
    unsigned m_reuse[MAX_MEMORY_ACCESS_SIZE+1];
    unsigned m_chunk_reuse[MAX_CACHE_CHUNK_NUM+1];
    unsigned m_chunk_cont_reuse[MAX_CACHE_CHUNK_NUM+1];
    unsigned m_issued_prefetch;
    unsigned m_useful_prefetch;
    unsigned m_useless_prefetch;
    unsigned m_miss_repl;
    unsigned m_partial_miss_repl;

    // performance counters for calculating the amount of misses within a time window
    unsigned m_prev_snapshot_access;
    unsigned m_prev_snapshot_miss;
    unsigned m_prev_snapshot_pending_hit;

    int m_core_id; // which shader core is using this
    int m_type_id; // what kind of cache is this (normal, texture, constant)

    // new fields
    LINE_REPLACEMENT_STATE   **repl;
    unsigned m_assoc;
    unsigned num_sets;

    unsigned m_bypasses;
    unsigned m_nonbypasses;
    unsigned * reused_num;
    unsigned * hitposition;
    unsigned m_hit_pos_max;

    cache_repl_type m_base_repl;
    unsigned * m_partition;
};

#define REUSED_MAX_NUM 16

// RRIP
#define RRPV_BITS 2
#define RRPV_DIS  (unsigned)((1 << RRPV_BITS) - 1)
#define RRPV_LONG (unsigned)((1 << RRPV_BITS) - 2)
