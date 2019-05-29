#include "../cuda-sim/cuda-sim.h"
#include "gpu-cache.h"
#include <iostream>
#include <sstream>
#include <assert.h>

using namespace std;

unsigned tag_array::get_victim(mem_fetch *mf, new_addr_type addr)
{
    assert(m_config.m_replacement_policy == PARTITION);

    int lruWay;
    unsigned victim_idx;
    new_addr_type victim_addr;
    unsigned set_index = m_config.set_index(addr);
    //unsigned tid = mf->get_wid(); //FIXME: how to generate tid?
    address_type PC = mf->get_pc();
    if(PC == (unsigned)-1) PC = addr;
    mem_access_type accessType = mf->get_access_type();
    unsigned cid = mf->get_inst().cta_id();

    assert(accessType != L2_WRBK_ACC && accessType != L2_WR_ALLOC_R);
    //bool is_real_access = (accessType != L1_WRBK_ACC && accessType != L1_WR_ALLOC_R &&
    //                       accessType != L1_PREF_ACC_R && accessType != L2_PREF_ACC_R &&
    //                       accessType != GLOBAL_ACC_W && accessType != LOCAL_ACC_W);
    if(accessType == L1_WRBK_ACC) // lld: these accesses should not affect performance
        return (unsigned)-1;

    assert(m_base_repl == BASE_REPL_LRU);
    if(m_base_repl == BASE_REPL_LRU)
        lruWay = Get_LRU_Victim(set_index,(unsigned)-1,cid);
    else if(m_base_repl == BASE_REPL_NRU)
        lruWay = Get_NRU_Victim(set_index,(unsigned)-1);
    else if(m_base_repl == BASE_REPL_SRRIP)
        lruWay = Get_RRIP_Victim(set_index,(unsigned)-1);
    else if(m_base_repl == BASE_REPL_RANDOM)
        lruWay = Get_Random_Victim(set_index,(unsigned)-1);
    else
        abort();

    if(lruWay == -1) {
        victim_idx = (unsigned)-1;
        victim_addr = addr;
    } else {
        victim_idx = set_index * m_assoc + lruWay;
        victim_addr = m_lines[victim_idx].m_block_addr;
    }

    //if(g_debug_execution >= 1 && m_core_id == 0)
    if(g_debug_execution >= 1) {
        stringstream ss; ss << "0x" << hex << victim_addr << " (" << lruWay << ")";
        string victim_status_str = (victim_idx == (unsigned)-1)?"NON":ss.str();
        cerr << "Core " << m_core_id << hex << " gets victim for 0x" << addr << ": " << victim_status_str << endl;
        cerr << dec << "\tcid=" << cid << ", tpc=" << mf->get_tpc() << hex << ", PC=0x" << PC << ", type=" << mem_access_type_str(accessType) << endl;
        fflush(stderr);
    }

    return victim_idx;
}

void tag_array::update(mem_fetch *mf, new_addr_type addr, unsigned idx, enum cache_request_status status)
{
    unsigned setIndex = m_config.set_index(addr);
    int updateWayID = idx % m_assoc;
    //unsigned tid = mf->get_wid(); //FIXME: how to generate tid?
    address_type PC = mf->get_pc();
    if(PC == (unsigned)-1) PC = addr;
    mem_access_type accessType = mf->get_access_type();
    unsigned cid = mf->get_inst().cta_id();

    assert(idx != (unsigned)-1 && accessType != L2_WRBK_ACC && accessType != L2_WR_ALLOC_R);
    bool is_real_access = (accessType != L1_WRBK_ACC && accessType != L1_WR_ALLOC_R &&
                           accessType != L1_PREF_ACC_R && accessType != L2_PREF_ACC_R &&
                           accessType != GLOBAL_ACC_W && accessType != LOCAL_ACC_W);
    //if(accessType == L1_WRBK_ACC || accessType == L1_WR_ALLOC_R || accessType == GLOBAL_ACC_W || accessType == LOCAL_ACC_W)
    if(accessType == L1_WRBK_ACC) // lld: these accesses should not affect performance
        return;

    if(!is_real_access)
        return;

    bool hit = (status == HIT) || (status == MISS_PARTIAL);

    if(hit && is_real_access) // there should be no L2 writeback
    {
        if(repl[setIndex][updateWayID].used_num < REUSED_MAX_NUM - 1)
        {
            reused_num[repl[setIndex][updateWayID].used_num]--;
            reused_num[repl[setIndex][updateWayID].used_num + 1]++;
        }
        repl[setIndex][updateWayID].used_num++;

        if(m_base_repl == BASE_REPL_LRU)
            hitposition[repl[setIndex][updateWayID].LRUstackposition]++;
        else if(m_base_repl == BASE_REPL_NRU)
        {
            if(repl[setIndex][updateWayID].NRUbit == true) hitposition[0]++;
            else hitposition[1]++;
        }
        else if(m_base_repl == BASE_REPL_SRRIP)
            hitposition[repl[setIndex][updateWayID].RRPV]++;
        else if(m_base_repl == BASE_REPL_PDP)
            hitposition[repl[setIndex][updateWayID].RPD]++;

        repl[setIndex][updateWayID].cid = cid;
    }
    else if(is_real_access)
    {
        repl[setIndex][updateWayID].used_num = 0;
        reused_num[0]++;
        repl[setIndex][updateWayID].cid = cid;
    }

    if(m_base_repl == BASE_REPL_LRU)
        UpdateLRU( setIndex, updateWayID );
    else if(m_base_repl == BASE_REPL_NRU)
        UpdateNRU( setIndex, updateWayID );
    else if(m_base_repl == BASE_REPL_SRRIP)
        UpdateSRRIP( setIndex, updateWayID, hit );
    else if(m_base_repl != BASE_REPL_RANDOM)
        abort();

    //if(g_debug_execution >= 1 && m_core_id == 0) {
    if(g_debug_execution >= 1) {
        cerr << dec << "Core " << m_core_id << hex << " updates 0x" << addr << " on " << cache_request_status_str(status) << endl;
        cerr << dec << "\tcid=" << cid << ", tpc=" << mf->get_tpc() << hex << ", PC=0x" << PC << ", type=" << mem_access_type_str(accessType) << endl;
        fflush(stderr);
    }
}

// check reservation fail
bool tag_array::check_avail(mem_fetch *mf, new_addr_type addr)
{
    //return true;
    unsigned setIndex = m_config.set_index(addr);
    unsigned cid = mf->get_inst().cta_id();
    // Get pointer to replacement state of current set
    LINE_REPLACEMENT_STATE *replSet = repl[ setIndex ];

    // calculate line number with the same cid
    int curnum = 0, curlimit;
    assert(cid < 48);
    if(m_partition[cid] == 0) {
        curlimit = m_partition[48];
        if(curlimit == 0)
            return true;
        for(unsigned way=0; way<m_assoc; way++) {
            unsigned idx = setIndex * m_assoc + way;
            if( m_lines[idx].m_status != INVALID && m_lines[idx].m_status != UNUSED && m_partition[replSet[way].cid] == 0 )
                curnum++;
        }
    
        if(curnum < curlimit) {
            for(unsigned way=0; way<m_assoc; way++) {
                unsigned idx = setIndex * m_assoc + way;
                if( m_lines[idx].m_status == INVALID )
                    return true;
                else if( m_partition[replSet[way].cid] > 0 && m_lines[idx].m_status != RESERVED && m_lines[idx].m_status != UNUSED )
                    return true;
            }
        } else {
            for(unsigned way=0; way<m_assoc; way++) {
                unsigned idx = setIndex * m_assoc + way;
                if( m_lines[idx].m_status != RESERVED && m_lines[idx].m_status != INVALID && m_lines[idx].m_status != UNUSED && m_partition[replSet[way].cid] == 0 )
                    return true;
            }
        }
    } else {
        curlimit = m_partition[cid];
        if(curlimit == 0)
            return true;
        for(unsigned way=0; way<m_assoc; way++) {
            unsigned idx = setIndex * m_assoc + way;
            if( m_lines[idx].m_status != INVALID && m_lines[idx].m_status != UNUSED && replSet[way].cid == cid )
                curnum++;
        }

        if(curnum < curlimit) {
            for(unsigned way=0; way<m_assoc; way++) {
                unsigned idx = setIndex * m_assoc + way;
                if( m_lines[idx].m_status == INVALID )
                    return true;
                else if( replSet[way].cid != cid && m_lines[idx].m_status != RESERVED && m_lines[idx].m_status != UNUSED )
                    return true;
            }
        } else {
            for(unsigned way=0; way<m_assoc; way++) {
                unsigned idx = setIndex * m_assoc + way;
                if( m_lines[idx].m_status != RESERVED && m_lines[idx].m_status != INVALID && m_lines[idx].m_status != UNUSED && replSet[way].cid == cid )
                    return true;
            }
        }
    }

    return false;
}

void tag_array::touch(mem_fetch *mf, new_addr_type addr)
{
}

// decide which parts should be allocated
void tag_array::decide_allocation(mem_fetch *mf, new_addr_type addr)
{
}

void tag_array::decide_prefetch(mem_fetch *mf, new_addr_type addr)
{
}

void tag_array::get_sub_stats(struct cache_sub_stats &css) const{
    struct cache_sub_stats t_css;
    t_css.clear();
    t_css.fills = m_fill;
    t_css.total_lat = m_total_lat;
    t_css.hit_fills = m_hit_fill;
    t_css.total_hit_lat = m_total_hit_lat;
    t_css.prefetches = m_issued_prefetch;
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
    for(unsigned i = 0; i < m_hit_pos_max-1; i++)
        out << i << "=" << hitposition[i] << ", ";
    out << m_hit_pos_max-1 << "=" << hitposition[m_hit_pos_max-1] << endl;
}

void tag_array::init_repl()
{
    m_base_repl = BASE_REPL_LRU;

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
            repl[ setIndex ][ way ].tid = 0;
            repl[ setIndex ][ way ].used_num = 0;
            repl[ setIndex ][ way ].cid = 0;
        }
    }

    if(m_base_repl == BASE_REPL_LRU)
        m_hit_pos_max = m_assoc;
    else if(m_base_repl == BASE_REPL_NRU)
        m_hit_pos_max = 2;
    else if(m_base_repl == BASE_REPL_SRRIP)
        m_hit_pos_max = RRPV_DIS + 1;
    else if(m_base_repl == BASE_REPL_RANDOM)
        m_hit_pos_max = 1;
    else
        m_hit_pos_max = -1;

    m_bypasses = 0;
    m_nonbypasses = 0;
    m_issued_prefetch = 0;
    m_useful_prefetch = 0;
    m_useless_prefetch = 0;
    reused_num = new unsigned[REUSED_MAX_NUM];
    assert(reused_num);
    for(int i = 0; i < REUSED_MAX_NUM; i++)
        reused_num[i] = 0;
    hitposition = new unsigned[m_hit_pos_max];
    assert(hitposition);
    for(unsigned i = 0; i < m_hit_pos_max; i++)
        hitposition[i] = 0;
    m_partition = new unsigned[49];
    assert(m_partition);
    for(unsigned i = 1; i < 48; i++)
        m_partition[i] = 0;
    m_partition[0] = 3;
    m_partition[1] = 1;
    m_partition[48] = 0;
}

unsigned tag_array::get_partial_victim(mem_fetch *mf, new_addr_type addr, unsigned idx)
{
    assert(0);
    return -1;
}

unsigned tag_array::get_real_victim(mem_fetch *mf, new_addr_type addr, unsigned idx)
{
    assert(0);
    return -1;
}

// Random
int tag_array::Get_Random_Victim( unsigned setIndex, unsigned used_idx )
{
    unsigned way;
    int avail_num = 0;

    for(way=0; way<m_assoc; way++) 
    {
        unsigned idx = setIndex * m_assoc + way;
        if(m_lines[idx].m_status != RESERVED && m_lines[idx].m_status != INVALID && m_lines[idx].m_status != UNUSED && way != used_idx)
            avail_num++;
    }
    assert(avail_num > 0);

    int lruWay = (rand() % avail_num);

    avail_num = 0;
    for(way=0; way<m_assoc; way++) 
    {
        unsigned idx = setIndex * m_assoc + way;
        if(m_lines[idx].m_status != RESERVED && m_lines[idx].m_status != INVALID && m_lines[idx].m_status != UNUSED && way != used_idx)
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
int tag_array::Get_LRU_Victim( unsigned setIndex, unsigned used_idx, unsigned cid )
{
    // Get pointer to replacement state of current set
    LINE_REPLACEMENT_STATE *replSet = repl[ setIndex ];

    int   lruWay   = -1;
    unsigned maxLRUstackposition = 0;

    // calculate line number with the same cid
    int curnum = 0, curlimit;
    assert(cid < 48);
    if(m_partition[cid] == 0) {
        curlimit = m_partition[48];
        if(curlimit == 0)
            return -1;
        for(unsigned way=0; way<m_assoc; way++) {
            unsigned idx = setIndex * m_assoc + way;
            if( m_lines[idx].m_status != INVALID && m_lines[idx].m_status != UNUSED && m_partition[replSet[way].cid] == 0 )
                curnum++;
        }
    
        if(curnum < curlimit) {
            for(unsigned way=0; way<m_assoc; way++) 
            {
                unsigned idx = setIndex * m_assoc + way;
                if( m_lines[idx].m_status == INVALID )
                    return way;
                else if( m_partition[replSet[way].cid] > 0 && replSet[way].LRUstackposition >= maxLRUstackposition && m_lines[idx].m_status != RESERVED && m_lines[idx].m_status != INVALID && m_lines[idx].m_status != UNUSED && way != used_idx )
                {
                    lruWay = way;
                    maxLRUstackposition = replSet[way].LRUstackposition;
                }
            }
        } else {
            for(unsigned way=0; way<m_assoc; way++) 
            {
                unsigned idx = setIndex * m_assoc + way;
                if( m_partition[replSet[way].cid] == 0 && replSet[way].LRUstackposition >= maxLRUstackposition && m_lines[idx].m_status != RESERVED && m_lines[idx].m_status != INVALID && m_lines[idx].m_status != UNUSED && way != used_idx )
                {
                    lruWay = way;
                    maxLRUstackposition = replSet[way].LRUstackposition;
                }
            }
        }
    } else {
        curlimit = m_partition[cid];
        if(curlimit == 0)
            return -1;
        for(unsigned way=0; way<m_assoc; way++) {
            unsigned idx = setIndex * m_assoc + way;
            if( m_lines[idx].m_status != INVALID && m_lines[idx].m_status != UNUSED && replSet[way].cid == cid )
                curnum++;
        }

        if(curnum < curlimit) {
            for(unsigned way=0; way<m_assoc; way++) 
            {
                unsigned idx = setIndex * m_assoc + way;
                if( m_lines[idx].m_status == INVALID )
                    return way;
                else if( replSet[way].cid != cid && replSet[way].LRUstackposition >= maxLRUstackposition && m_lines[idx].m_status != RESERVED && m_lines[idx].m_status != INVALID && m_lines[idx].m_status != UNUSED && way != used_idx )
                {
                    lruWay = way;
                    maxLRUstackposition = replSet[way].LRUstackposition;
                }
            }
        } else {
            for(unsigned way=0; way<m_assoc; way++) 
            {
                unsigned idx = setIndex * m_assoc + way;
                if( replSet[way].cid == cid && replSet[way].LRUstackposition >= maxLRUstackposition && m_lines[idx].m_status != RESERVED && m_lines[idx].m_status != INVALID && m_lines[idx].m_status != UNUSED && way != used_idx )
                {
                    lruWay = way;
                    maxLRUstackposition = replSet[way].LRUstackposition;
                }
            }
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
int tag_array::Get_NRU_Victim( unsigned setIndex, unsigned used_idx )
{
    // Get pointer to replacement state of current set
    LINE_REPLACEMENT_STATE *replSet = repl[ setIndex ];

    // Search for victim whose stack position is m_assoc-1
    for(unsigned way=0; way<m_assoc; way++) 
    {
        unsigned idx = setIndex * m_assoc + way;
        if( replSet[way].NRUbit == false && m_lines[idx].m_status != RESERVED && m_lines[idx].m_status != INVALID && m_lines[idx].m_status != UNUSED && way != used_idx )
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
int tag_array::Get_RRIP_Victim( unsigned setIndex, unsigned used_idx )
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
        if(replSet[way].RRPV >= rrpv_max && m_lines[idx].m_status != RESERVED && m_lines[idx].m_status != INVALID && m_lines[idx].m_status != UNUSED && way != used_idx)
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
