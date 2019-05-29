//#define REPL_DEBUG

enum cache_repl_type {
    BASE_REPL_LRU = 0,
    BASE_REPL_NRU,
    BASE_REPL_SRRIP,
    BASE_REPL_RANDOM,
    BASE_REPL_PDP
};

struct BM_entry {
    bool valid;
    bool last_itag;
    unsigned Itid;
    unsigned Itype;
    new_addr_type Ipc;
    int Isign;
    new_addr_type Itag;
    new_addr_type Vtag;
    std::bitset<MAX_CACHE_CHUNK_NUM> i_sector_mask;
    std::bitset<MAX_CACHE_CHUNK_NUM> v_sector_mask;
    unsigned  LRUstackposition;
};

class BypassMonitor
{
  private:
    // parameters
    unsigned numsets;
    unsigned assoc;
    unsigned m_update_pr;
    unsigned m_bd_addr_len;
    unsigned m_bd_max_addr;
    unsigned m_bd_nbit;
    char m_bd_idx;
    unsigned m_bm_partial_tag_nbit;
    cache_config m_cache_config;

    BM_entry **entries;
    int** BypassDecider;

    // structure parameters
    unsigned linesize;
    unsigned lineShift;
    unsigned indexShift;
    unsigned indexMask;

    //status
    unsigned* access_num;
    unsigned* hititag_num;
    unsigned* hitvtag_num;
    unsigned* evict_num;
    unsigned* evicted_num;
    unsigned* vichititag_num;
    unsigned* BD_saturate_counter;

    void UpdateLRU( unsigned setIndex, int updateWayID );
    void UpdateBD(unsigned tid, new_addr_type PC, new_addr_type addr, new_addr_type vaddr, int sign, bool direction, unsigned weight, unsigned accessType);
    new_addr_type GetBDaddr(unsigned &tid, new_addr_type PC, new_addr_type addr, new_addr_type vaddr, int sign, unsigned accessType);

    new_addr_type GetTag( new_addr_type addr );
    new_addr_type GetAddr(unsigned setIndex, new_addr_type tag)
    {
        if(m_cache_config.m_set_index_function == FERMI_HASH_SET_FUNCTION)
            return (tag << lineShift);
        else
            return (((tag << indexShift) + setIndex) << lineShift);
    }
    int SaturateAdd(int x, int inc, int max);
    int SaturateSub(int x, int inc, int max);

  public:
    cache_repl_type m_base_repl;
    bool m_prefetch_aware;
    unsigned m_rrpv_nbit;
    unsigned NumThreadsPerCache;  // number threads per cache
    unsigned misses;

    BypassMonitor(cache_config cfg);
    int  Get_Victim( unsigned tid, unsigned setIndex );
    void Fill(int set, int way, mem_fetch *mf, unsigned tid, address_type pc, new_addr_type addr, cache_block_t *vic_blk, int sign, mem_access_type type, bool l2);
    int  LookUp(mem_fetch *mf, unsigned tid, new_addr_type addr, bool l2);
    void LookUpVictim(unsigned tid, cache_block_t *vic_blk);
    bool IsBypass(unsigned tid, new_addr_type PC, new_addr_type addr, new_addr_type vaddr, int sign, unsigned accessType);
    unsigned GetSetIndex( new_addr_type addr )
    {
        if(m_cache_config.m_set_index_function == FERMI_HASH_SET_FUNCTION) {
            assert(numsets <= 32);
            assert(linesize == 128);

            unsigned lower_xor = 0;
            unsigned upper_xor = 0;
            lower_xor = (addr >> lineShift) & 0x1F;
            // Upper xor value is bits 13, 14, 15, 17, and 19
            upper_xor  = (addr & 0xE000)  >> 13; // Bits 13, 14, 15
            upper_xor |= (addr & 0x20000) >> 14; // Bit 17
            upper_xor |= (addr & 0x80000) >> 15; // Bit 19

            return (lower_xor ^ upper_xor);
            //return (m_cache_config.set_index(addr) & indexMask);
        } else
            return ((addr >> lineShift) & indexMask);
    }

    void Print(std::ostream &out);
    void PrintStats(std::ostream &out);
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

} LINE_REPLACEMENT_STATE;

#define CACHE_REPL_STATS_TUP_DEF \
CRS_TUP_BEGIN( cache_repl_stats ) \
   CRS_TUP( INTRA_HIT ), \
   CRS_TUP( INTER_HIT ), \
   CRS_TUP( BYPASS ), \
   CRS_TUP( NONBYPASS ), \
   CRS_TUP( EXTRAREPL ), \
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
    unsigned hash_function(int idx, unsigned key); // hash function for LAMAR

    // new functions
    int  Get_Random_Victim( unsigned setIndex, unsigned used_idx );
    int  Get_LRU_Victim( unsigned setIndex, unsigned used_idx );
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
    unsigned m_extra_repl;
    unsigned * reused_num;
    unsigned * hitposition;
    unsigned m_hit_pos_max;

    BypassMonitor* BM;
};

#define REUSED_MAX_NUM 16

// Bypass Decider Parameters
#define BD_ADDR_MASK ((1 << m_bd_addr_len) - 1)
#define BD_PC_SHIFT 3
#define BD_ADDR_SHIFT 10  // Sub 2 for multi-thread workloads

#define BD_BIT m_bd_nbit
#define BD_MAX ((1 << (BD_BIT - 1)) - 1)

#define BM_PARTIAL_TAG_BIT m_bm_partial_tag_nbit
#define BM_PARTIAL_TAG_MASK ((1 << BM_PARTIAL_TAG_BIT) - 1)

// RRIP
#define RRPV_BITS (BM->m_rrpv_nbit)
#define RRPV_DIS  (unsigned)((1 << RRPV_BITS) - 1)
#define RRPV_LONG (unsigned)((1 << RRPV_BITS) - 2)
