// Baseline Replacement Policy Types
#define BASE_REPL_LRU
//#define BASE_REPL_NRU
//#define BASE_REPL_SRRIP
//#define BASE_REPL_PDP
//#define BASE_REPL_RANDOM
//#define BASE_REPL_BYPASS

#define THREAD_NUM 1

// Replacement State Per Cache Line
typedef struct
{
    unsigned LRUstackposition;
    bool NRUbit;
    unsigned RRPV;
    unsigned RPD; // remaining PD
    unsigned tid;
    unsigned used_num;

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

    enum cache_request_status probe_selector( mem_fetch *mf, new_addr_type addr, unsigned &idx, std::list<unsigned> &repl_idx, unsigned time );
    enum cache_request_status access_probe( mem_fetch *mf, new_addr_type addr, unsigned &idx, std::list<unsigned> &repl_idx, unsigned time );
    unsigned get_victim(mem_fetch *mf, new_addr_type addr);
    void update(mem_fetch *mf, new_addr_type addr, unsigned idx, bool hit);

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
    int  Get_Random_Victim( unsigned setIndex );
    int  Get_LRU_Victim( unsigned setIndex );
    void UpdateLRU( unsigned setIndex, int updateWayID );
    int  Get_NRU_Victim( unsigned setIndex );
    void UpdateNRU( unsigned setIndex, int updateWayID );
    int  Get_RRIP_Victim( unsigned setIndex );
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
};

// RRIP
#define RRPV_BITS 2
#define RRPV_DIS  ((1 << RRPV_BITS) - 1)
#define RRPV_LONG ((1 << RRPV_BITS) - 2)

#define REUSED_MAX_NUM 16
#if defined(BASE_REPL_LRU)
    #define HIT_POS_MAX m_assoc
#elif defined(BASE_REPL_NRU)
    #define HIT_POS_MAX 2
#elif defined(BASE_REPL_SRRIP)
    #define HIT_POS_MAX (RRPV_DIS + 1)
#elif defined(BASE_REPL_PDP)
    #define HIT_POS_MAX 1
#elif defined(BASE_REPL_RANDOM)
    #define HIT_POS_MAX 1
#else
    #define HIT_POS_MAX 1
#endif
