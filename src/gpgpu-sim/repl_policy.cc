#include "../cuda-sim/cuda-sim.h"
#include "gpu-cache.h"
#include <iostream>
#include <sstream>
#include <assert.h>

using namespace std;

unsigned tag_array::get_victim(mem_fetch *mf, new_addr_type addr)
{
    int lruWay;
    unsigned victim_idx;
    new_addr_type victim_addr = 0;
    unsigned set_index = m_config.set_index(addr);
    unsigned tid = mf->get_wid();
    address_type PC = mf->get_pc();
    if(PC == (unsigned)-1) PC = addr;
    mem_access_type accessType = mf->get_access_type();

#if defined(BASE_REPL_LRU)
    lruWay = Get_LRU_Victim(set_index);
#elif defined(BASE_REPL_NRU)
    lruWay = Get_NRU_Victim(set_index);
#elif defined(BASE_REPL_SRRIP)
    lruWay = Get_RRIP_Victim(set_index);
#elif defined(BASE_REPL_PDP)
    abort();
#elif defined(BASE_REPL_RANDOM)
    lruWay = Get_Random_Victim(set_index);
#elif defined(BASE_REPL_BYPASS)
    lruWay = -1;
#else
    abort();
#endif

    if(lruWay >= 0)
    {
        victim_idx = set_index * m_assoc + lruWay;
        victim_addr = m_lines[victim_idx].m_block_addr;
        m_nonbypasses++;
    }
    else
    {
        victim_idx = (unsigned)-1;
        m_bypasses++;
    }

    //if(g_debug_execution >= 1 && m_core_id == 0)
    if(g_debug_execution >= 1) {
        stringstream ss; ss << "0x" << hex << victim_addr << " (" << lruWay << ")";
        string victim_status_str = (victim_idx == (unsigned)-1)?"NON":ss.str();
        cerr << "Core " << m_core_id << hex << " gets victim for 0x" << addr << ": " << victim_status_str << endl;
        cerr << dec << "\twid=" << tid << ", tpc=" << mf->get_tpc() << hex << ", PC=0x" << PC << ", type=" << mem_access_type_str(accessType) << endl;
        fflush(stderr);
    }

    return victim_idx;
}

void tag_array::update(mem_fetch *mf, new_addr_type addr, unsigned idx, bool hit)
{
    assert(idx != (unsigned)-1);

    unsigned setIndex = m_config.set_index(addr);
    int updateWayID = idx % m_assoc;
    unsigned tid = mf->get_wid();
    address_type PC = mf->get_pc();
    if(PC == (unsigned)-1) PC = addr;
    mem_access_type accessType = mf->get_access_type();

    assert(accessType != L2_WRBK_ACC);
    if(accessType == L1_WRBK_ACC)
        return;

    if(accessType == L1_PREF_ACC_R || accessType == L2_PREF_ACC_R ||
       accessType == L1_WR_ALLOC_R || accessType == L2_WR_ALLOC_R ||
       accessType == GLOBAL_ACC_W || accessType == LOCAL_ACC_W) // lld: these accesses should not affect performance
        return;

    if(hit && accessType != L1_WRBK_ACC) // there should be no L2 writeback
    {
        if(repl[setIndex][updateWayID].used_num < REUSED_MAX_NUM - 1)
        {
            reused_num[repl[setIndex][updateWayID].used_num]--;
            reused_num[repl[setIndex][updateWayID].used_num + 1]++;
        }
        repl[setIndex][updateWayID].used_num++;

#if defined(BASE_REPL_LRU)
        hitposition[repl[setIndex][updateWayID].LRUstackposition]++;
#elif defined(BASE_REPL_NRU)
        if(repl[setIndex][updateWayID].NRUbit == true) hitposition[0]++;
        else hitposition[1]++;
#elif defined(BASE_REPL_SRRIP)
        hitposition[repl[setIndex][updateWayID].RRPV]++;
#elif defined(BASE_REPL_PDP)
        hitposition[repl[setIndex][updateWayID].RPD]++;
#else
#endif
    }
    else
    {
        repl[setIndex][updateWayID].used_num = 0;
        reused_num[0]++;
    }


#if defined(BASE_REPL_LRU)
    UpdateLRU( setIndex, updateWayID );
#elif defined(BASE_REPL_NRU)
    UpdateNRU( setIndex, updateWayID );
#elif defined(BASE_REPL_SRRIP)
    UpdateSRRIP( setIndex, updateWayID, hit );
#elif defined(BASE_REPL_PDP)
    abort();
#elif defined(BASE_REPL_RANDOM)
#elif defined(BASE_REPL_BYPASS)
#else
    abort();
#endif

    //if(g_debug_execution >= 1 && m_core_id == 0) {
    if(g_debug_execution >= 1) {
        string access_status_str = hit?"hit":"miss";
        cerr << dec << "Core " << m_core_id << hex << " updates 0x" << addr << " on " << access_status_str << endl;
        cerr << dec << "\twid=" << tid << ", tpc=" << mf->get_tpc() << hex << ", PC=0x" << PC << ", type=" << mem_access_type_str(accessType) << endl;
        fflush(stderr);
    }
}

void tag_array::get_sub_stats(struct cache_sub_stats &css) const{
    struct cache_sub_stats t_css;
    t_css.clear();
    t_css.fills = m_fill;
    t_css.total_lat = m_total_lat;
    t_css.hit_fills = m_hit_fill;
    t_css.total_hit_lat = m_total_hit_lat;
    t_css.useful_prefetches = m_useful_prefetch;
    t_css.useless_prefetches = m_useless_prefetch;
    t_css.miss_repl = m_miss_repl;
    t_css.partial_miss_repl = m_partial_miss_repl;
    for(unsigned i = 0; i <= MAX_MEMORY_ACCESS_SIZE; i++)
        t_css.reuse[i] = m_reuse[i];
    for(unsigned i = 0; i <= MAX_CACHE_CHUNK_NUM; i++) {
        t_css.chunk_reuse[i] = m_chunk_reuse[i];
        t_css.chunk_cont_reuse[i] = m_chunk_cont_reuse[i];
    }
    t_css.repl_stats[INTRA_HIT] = m_intra_warp_hit;
    t_css.repl_stats[INTER_HIT] = m_inter_warp_hit;

    if( m_config.m_custom_repl != 1 )
    {
        css += t_css;
        return;
    }

    t_css.repl_stats[BYPASS] = m_bypasses;
    t_css.repl_stats[NONBYPASS] = m_nonbypasses;
    
    css += t_css;
}

void tag_array::print_stats( ostream &out, string name ) const
{
    out << name << ":\t" << "IntraHit = " << m_intra_warp_hit << ", InterHit = " << m_inter_warp_hit
                         << ", bypasses = " << m_bypasses << ", nonbypasses = " << m_nonbypasses << endl;

    // reused number
    out << "\t" << "reuse stat: ";
    unsigned total = 0;
    for(int i = 0; i < REUSED_MAX_NUM - 1; i++)
    {
        out << i << "=" << reused_num[i] << ", ";
        total += reused_num[i];
    }
    out << "above " << REUSED_MAX_NUM - 2 << "=" << reused_num[REUSED_MAX_NUM - 1] << ". ";
    total += reused_num[REUSED_MAX_NUM - 1];
    out << "zero reused rate: " << 100.0 * (double)reused_num[0] / (double)total << endl;
    // hit position
    out << "\t" << "hit position: ";
    for(unsigned i = 0; i < HIT_POS_MAX-1; i++)
        out << i << "=" << hitposition[i] << ", ";
    out << HIT_POS_MAX-1 << "=" << hitposition[HIT_POS_MAX-1] << endl;
}

void tag_array::init_repl()
{
    num_sets = m_config.m_nset;
    m_assoc = m_config.m_assoc;

    // Create the state for sets, then create the state for the ways
    repl  = new LINE_REPLACEMENT_STATE* [ num_sets ];

    // ensure that we were able to create replacement state
    assert(repl);

    // Create the state for the sets
    for(unsigned setIndex=0; setIndex<num_sets; setIndex++) 
    {
        repl[ setIndex ]  = new LINE_REPLACEMENT_STATE[ m_assoc ];
        assert(repl[ setIndex ]);

        for(unsigned way=0; way<m_assoc; way++) 
        {
            // initialize stack position (for true LRU)
            repl[ setIndex ][ way ].LRUstackposition = way;
            repl[ setIndex ][ way ].NRUbit = false;
            repl[ setIndex ][ way ].RRPV = RRPV_DIS;
            repl[ setIndex ][ way ].tid = THREAD_NUM;
            repl[ setIndex ][ way ].used_num = 0;
        }
    }

    m_bypasses = 0;
    m_nonbypasses = 0;
    reused_num = new unsigned[REUSED_MAX_NUM];
    assert(reused_num);
    for(int i = 0; i < REUSED_MAX_NUM; i++)
        reused_num[i] = 0;
    hitposition = new unsigned[HIT_POS_MAX];
    assert(hitposition);
    for(unsigned i = 0; i < HIT_POS_MAX; i++)
        hitposition[i] = 0;
}

// Random
int tag_array::Get_Random_Victim( unsigned setIndex )
{
    unsigned way;
    int avail_num = 0;

    for(way=0; way<m_assoc; way++) 
    {
        unsigned idx = setIndex * m_assoc + way;
        if(m_lines[idx].m_status != RESERVED)
            avail_num++;
    }
    assert(avail_num > 0);

    int lruWay = (rand() % avail_num);

    avail_num = 0;
    for(way=0; way<m_assoc; way++) 
    {
        unsigned idx = setIndex * m_assoc + way;
        if(m_lines[idx].m_status != RESERVED)
        {
            if(avail_num == lruWay)
                return way;
            avail_num++;
        }
    }
    
    abort();
    return -1;
}

// LRU
int tag_array::Get_LRU_Victim( unsigned setIndex )
{
    // Get pointer to replacement state of current set
    LINE_REPLACEMENT_STATE *replSet = repl[ setIndex ];

    int   lruWay   = -1;
    unsigned maxLRUstackposition = 0;

    // Search for victim whose stack position is m_assoc-1
    for(unsigned way=0; way<m_assoc; way++) 
    {
        unsigned idx = setIndex * m_assoc + way;
        if( replSet[way].LRUstackposition >= maxLRUstackposition && m_lines[idx].m_status != RESERVED ) 
        {
            lruWay = way;
            maxLRUstackposition = replSet[way].LRUstackposition;
        }
    }

    // return lru way
    assert(lruWay >= 0);
    return lruWay;
}

void tag_array::UpdateLRU( unsigned setIndex, int updateWayID )
{
    // Determine current LRU stack position
    unsigned currLRUstackposition = repl[ setIndex ][ updateWayID ].LRUstackposition;

    // Update the stack position of all lines before the current line
    // Update implies incremeting their stack positions by one
    for(unsigned way=0; way<m_assoc; way++) 
    {
        if( repl[setIndex][way].LRUstackposition < currLRUstackposition ) 
            repl[setIndex][way].LRUstackposition++;
    }

    // Set the LRU stack position of new line to be zero
    repl[ setIndex ][ updateWayID ].LRUstackposition = 0;
}

// NRU
int tag_array::Get_NRU_Victim( unsigned setIndex )
{
    // Get pointer to replacement state of current set
    LINE_REPLACEMENT_STATE *replSet = repl[ setIndex ];

    // Search for victim whose stack position is m_assoc-1
    for(unsigned way=0; way<m_assoc; way++) 
    {
        unsigned idx = setIndex * m_assoc + way;
        if( replSet[way].NRUbit == false && m_lines[idx].m_status != RESERVED ) 
            return way;
    }

    for(unsigned way=0; way<m_assoc; way++) 
    {
        replSet[way].NRUbit = false;
        unsigned idx = setIndex * m_assoc + way;
        if( m_lines[idx].m_status != RESERVED ) 
            return way;
    }

    abort();
    return 0;
}

void tag_array::UpdateNRU( unsigned setIndex, int updateWayID )
{
    // Set the NRU bit of new line to be one
    repl[ setIndex ][ updateWayID ].NRUbit = true;
}

// SRRIP
int tag_array::Get_RRIP_Victim( unsigned setIndex )
{
    // Get pointer to replacement state of current set
    LINE_REPLACEMENT_STATE *replSet = repl[ setIndex ];

    int   replway   = -1;
    unsigned way;
    unsigned rrpv_max = 0;

    for(way=0; way<m_assoc; way++) 
    {
        if(replSet[way].RRPV == RRPV_DIS) 
            break;

        if(replSet[way].RRPV > rrpv_max)
            rrpv_max = replSet[way].RRPV;
    }
    if(way == m_assoc)
    {
        for(way=0; way<m_assoc; way++) 
            replSet[way].RRPV += (RRPV_DIS - rrpv_max);
    }

    rrpv_max = 0;
    for(way=m_assoc-1; way>=0; way--) 
    {
        unsigned idx = setIndex * m_assoc + way;
        if(replSet[way].RRPV >= rrpv_max && m_lines[idx].m_status != RESERVED) 
        {
            replway = way;
            rrpv_max = replSet[way].RRPV;
        }
        if(way == 0)
            break;
    }

    assert(replway >= 0);
    return replway;
}

void tag_array::UpdateSRRIP( unsigned setIndex, int updateWayID, bool cacheHit )
{
    assert(updateWayID != -1);

    if(cacheHit == true)
        repl[setIndex][updateWayID].RRPV = 0;
    else  //miss
        repl[setIndex][updateWayID].RRPV = RRPV_LONG;
}
