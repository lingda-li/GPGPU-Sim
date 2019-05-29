#include "../cuda-sim/cuda-sim.h"
#include "gpu-cache.h"
#include <iostream>
#include <sstream>
#include <assert.h>

using namespace std;

// decide which parts should be allocated
void tag_array::decide_allocation(mem_fetch *mf, new_addr_type addr)
{
}

void tag_array::decide_prefetch(mem_fetch *mf, new_addr_type addr)
{
}

unsigned tag_array::get_victim(mem_fetch *mf, new_addr_type addr)
{
    assert(m_config.m_replacement_policy == MY_REPL);

    int lruWay;
    unsigned victim_idx;
    new_addr_type victim_addr;
    unsigned set_index = m_config.set_index(addr);
    unsigned tid = mf->get_wid() % BM->NumThreadsPerCache; //FIXME: how to generate tid?
    address_type PC = mf->get_pc();
    if(PC == (unsigned)-1) PC = addr;
    mem_access_type accessType = mf->get_access_type();

    assert(accessType != L2_WRBK_ACC && accessType != L2_WR_ALLOC_R);
    //bool is_real_access = (accessType != L1_WRBK_ACC && accessType != L1_WR_ALLOC_R &&
    //                       accessType != L1_PREF_ACC_R && accessType != L2_PREF_ACC_R &&
    //                       accessType != GLOBAL_ACC_W && accessType != LOCAL_ACC_W);
    if(accessType == L1_WRBK_ACC) // lld: these accesses should not affect performance
        return (unsigned)-1;
    BM->misses++;

    if(BM->m_base_repl == BASE_REPL_LRU)
        lruWay = Get_LRU_Victim(set_index,(unsigned)-1);
    else if(BM->m_base_repl == BASE_REPL_NRU)
        lruWay = Get_NRU_Victim(set_index,(unsigned)-1);
    else if(BM->m_base_repl == BASE_REPL_SRRIP)
        lruWay = Get_RRIP_Victim(set_index,(unsigned)-1);
    else if(BM->m_base_repl == BASE_REPL_RANDOM)
        lruWay = Get_Random_Victim(set_index,(unsigned)-1);
    else
        abort();

    victim_idx = set_index * m_assoc + lruWay;
    victim_addr = m_lines[victim_idx].m_block_addr;
    BM->LookUpVictim(tid, victim_addr);

    // Update Bypass Monitor
    int BM_victim_set = BM->GetSetIndex(addr); //FIXME: should not use hash-index
    int BM_victim_way = BM->Get_Victim(tid, BM_victim_set);
    if(BM_victim_way >= 0)
        BM->Fill(BM_victim_set, BM_victim_way, tid, PC, addr, victim_addr, mf->get_sign(), accessType);

    // Make bypassing decision
    if(BM->IsBypass(tid, PC, addr, victim_addr, mf->get_sign(), accessType))
    {
        victim_idx = (unsigned)-1;
        m_bypasses++;
    }
    else
        m_nonbypasses++;

    //if(g_debug_execution >= 1 && m_core_id == 0)
    if(g_debug_execution >= 1) {
        stringstream ss; ss << "0x" << hex << victim_addr << " (" << lruWay << ")";
        string victim_status_str = (victim_idx == (unsigned)-1)?"NON":ss.str();
        cerr << "Core " << m_core_id << hex << " gets victim for 0x" << addr << ": " << victim_status_str << endl;
        cerr << dec << "\twid=" << tid << ", tpc=" << mf->get_tpc() << hex << ", PC=0x" << PC << ", type=" << mem_access_type_str(accessType) << endl;
#ifdef REPL_DEBUG
        BM->Print(cerr);
#endif
        fflush(stderr);
    }

    return victim_idx;
}

void tag_array::update(mem_fetch *mf, new_addr_type addr, unsigned idx, enum cache_request_status status)
{
    unsigned setIndex = m_config.set_index(addr);
    int updateWayID = idx % m_assoc;
    unsigned tid = mf->get_wid() % BM->NumThreadsPerCache; //FIXME: how to generate tid?
    address_type PC = mf->get_pc();
    if(PC == (unsigned)-1) PC = addr;
    mem_access_type accessType = mf->get_access_type();

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

        if(BM->m_base_repl == BASE_REPL_LRU)
            hitposition[repl[setIndex][updateWayID].LRUstackposition]++;
        else if(BM->m_base_repl == BASE_REPL_NRU)
        {
            if(repl[setIndex][updateWayID].NRUbit == true) hitposition[0]++;
            else hitposition[1]++;
        }
        else if(BM->m_base_repl == BASE_REPL_SRRIP)
            hitposition[repl[setIndex][updateWayID].RRPV]++;
        else if(BM->m_base_repl == BASE_REPL_PDP)
            hitposition[repl[setIndex][updateWayID].RPD]++;
    }
    else if(is_real_access)
    {
        repl[setIndex][updateWayID].used_num = 0;
        reused_num[0]++;
    }

    if(BM->m_base_repl == BASE_REPL_LRU)
        UpdateLRU( setIndex, updateWayID );
    else if(BM->m_base_repl == BASE_REPL_NRU)
        UpdateNRU( setIndex, updateWayID );
    else if(BM->m_base_repl == BASE_REPL_SRRIP)
        UpdateSRRIP( setIndex, updateWayID, hit );
    else if(BM->m_base_repl != BASE_REPL_RANDOM)
        abort();

    //if(g_debug_execution >= 1 && m_core_id == 0) {
    if(g_debug_execution >= 1) {
        cerr << dec << "Core " << m_core_id << hex << " updates 0x" << addr << " on " << cache_request_status_str(status) << endl;
        cerr << dec << "\twid=" << tid << ", tpc=" << mf->get_tpc() << hex << ", PC=0x" << PC << ", type=" << mem_access_type_str(accessType) << endl;
        fflush(stderr);
    }
}

void tag_array::touch(mem_fetch *mf, new_addr_type addr)
{
    unsigned tid = mf->get_wid() % BM->NumThreadsPerCache; //FIXME: how to generate tid?
    address_type PC = mf->get_pc();
    if(PC == (unsigned)-1) PC = addr;
    mem_access_type accessType = mf->get_access_type();

    assert(accessType != L2_WRBK_ACC && accessType != L2_WR_ALLOC_R);
    bool is_real_access = (accessType != L1_WRBK_ACC && accessType != L1_WR_ALLOC_R &&
                           accessType != L1_PREF_ACC_R && accessType != L2_PREF_ACC_R &&
                           accessType != GLOBAL_ACC_W && accessType != LOCAL_ACC_W);
    if(!is_real_access) // lld: these accesses should not affect performance
        return;

    BM->LookUp(tid, addr); // Check Bypass Monitor

    //if(g_debug_execution >= 1 && m_core_id == 0) {
    if(g_debug_execution >= 1) {
        cerr << dec << "Core " << m_core_id << hex << " touches 0x" << addr << ":" << endl;
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
    t_css.repl_stats[EXTRAREPL] = m_extra_repl;
    
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

    BM->PrintStats(out);
}

void tag_array::init_repl()
{
    BM = new BypassMonitor(m_config);

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
        }
    }

    if(BM->m_base_repl == BASE_REPL_LRU)
        m_hit_pos_max = m_assoc;
    else if(BM->m_base_repl == BASE_REPL_NRU)
        m_hit_pos_max = 2;
    else if(BM->m_base_repl == BASE_REPL_SRRIP)
        m_hit_pos_max = RRPV_DIS + 1;
    else if(BM->m_base_repl == BASE_REPL_RANDOM)
        m_hit_pos_max = 1;
    else
        m_hit_pos_max = -1;

    m_bypasses = 0;
    m_nonbypasses = 0;
    m_issued_prefetch = 0;
    m_useful_prefetch = 0;
    m_useless_prefetch = 0;
    m_extra_repl = 0;
    reused_num = new unsigned[REUSED_MAX_NUM];
    assert(reused_num);
    for(int i = 0; i < REUSED_MAX_NUM; i++)
        reused_num[i] = 0;
    hitposition = new unsigned[m_hit_pos_max];
    assert(hitposition);
    for(unsigned i = 0; i < m_hit_pos_max; i++)
        hitposition[i] = 0;
}

unsigned tag_array::get_partial_victim(mem_fetch *mf, new_addr_type addr, unsigned idx)
{
    int lruWay;
    unsigned victim_idx;
    new_addr_type victim_addr;
    unsigned set_index = m_config.set_index(addr);
    unsigned tid = mf->get_wid() % BM->NumThreadsPerCache; //FIXME: how to generate tid?
    address_type PC = mf->get_pc();
    if(PC == (unsigned)-1) PC = addr;
    mem_access_type accessType = mf->get_access_type();

    assert(accessType != L2_WRBK_ACC && accessType != L2_WR_ALLOC_R);
    //bool is_real_access = (accessType != L1_WRBK_ACC && accessType != L1_WR_ALLOC_R &&
    //                       accessType != L1_PREF_ACC_R && accessType != L2_PREF_ACC_R &&
    //                       accessType != GLOBAL_ACC_W && accessType != LOCAL_ACC_W);
    if(accessType == L1_WRBK_ACC) // lld: these accesses should not affect performance
        return (unsigned)-1;
    BM->misses++;

    if(BM->m_base_repl == BASE_REPL_LRU)
        lruWay = Get_LRU_Victim(set_index,idx);
    else if(BM->m_base_repl == BASE_REPL_NRU)
        lruWay = Get_NRU_Victim(set_index,idx);
    else if(BM->m_base_repl == BASE_REPL_SRRIP)
        lruWay = Get_RRIP_Victim(set_index,idx);
    else if(BM->m_base_repl == BASE_REPL_RANDOM)
        lruWay = Get_Random_Victim(set_index,idx);
    else
        abort();

    victim_idx = set_index * m_assoc + lruWay;
    victim_addr = m_lines[victim_idx].m_block_addr;
    BM->LookUpVictim(tid, victim_addr);

    // Update Bypass Monitor
    int BM_victim_set = BM->GetSetIndex(addr); //FIXME: should not use hash-index
    int BM_victim_way = BM->Get_Victim(tid, BM_victim_set);
    if(BM_victim_way >= 0)
        BM->Fill(BM_victim_set, BM_victim_way, tid, PC, addr, victim_addr, mf->get_sign(), accessType);

    // Make bypassing decision
    if(BM->IsBypass(tid, PC, addr, victim_addr, mf->get_sign(), accessType))
    {
        victim_idx = (unsigned)-1;
        m_bypasses++;
    }
    else
        m_nonbypasses++;

    //if(g_debug_execution >= 1 && m_core_id == 0)
    if(g_debug_execution >= 1) {
        stringstream ss; ss << "0x" << hex << victim_addr << " (" << lruWay << ")";
        string victim_status_str = (victim_idx == (unsigned)-1)?"NON":ss.str();
        cerr << "Core " << m_core_id << hex << " gets victim for 0x" << addr << ": " << victim_status_str << endl;
        cerr << dec << "\twid=" << tid << ", tpc=" << mf->get_tpc() << hex << ", PC=0x" << PC << ", type=" << mem_access_type_str(accessType) << endl;
#ifdef REPL_DEBUG
        BM->Print(cerr);
#endif
        fflush(stderr);
    }

    return victim_idx;
}

unsigned tag_array::get_real_victim(mem_fetch *mf, new_addr_type addr, unsigned idx)
{
    int lruWay;
    unsigned victim_idx;
    new_addr_type victim_addr;
    unsigned set_index = m_config.set_index(addr);
    unsigned tid = mf->get_wid() % BM->NumThreadsPerCache; //FIXME: how to generate tid?
    address_type PC = mf->get_pc();
    if(PC == (unsigned)-1) PC = addr;
    mem_access_type accessType = mf->get_access_type();

    assert(accessType != L2_WRBK_ACC && accessType != L2_WR_ALLOC_R);
    assert(accessType != L1_WRBK_ACC);
    m_extra_repl++;

    if(BM->m_base_repl == BASE_REPL_LRU)
        lruWay = Get_LRU_Victim(set_index,idx);
    else if(BM->m_base_repl == BASE_REPL_NRU)
        lruWay = Get_NRU_Victim(set_index,idx);
    else if(BM->m_base_repl == BASE_REPL_SRRIP)
        lruWay = Get_RRIP_Victim(set_index,idx);
    else if(BM->m_base_repl == BASE_REPL_RANDOM)
        lruWay = Get_Random_Victim(set_index,idx);
    else
        abort();

    victim_idx = set_index * m_assoc + lruWay;
    victim_addr = m_lines[victim_idx].m_block_addr;
    BM->LookUpVictim(tid, victim_addr);

    //if(g_debug_execution >= 1 && m_core_id == 0)
    if(g_debug_execution >= 1) {
        stringstream ss; ss << "0x" << hex << victim_addr << " (" << lruWay << ")";
        string victim_status_str = (victim_idx == (unsigned)-1)?"NON":ss.str();
        cerr << "Core " << m_core_id << hex << " gets extra victim for 0x" << addr << ": " << victim_status_str << endl;
        cerr << dec << "\twid=" << tid << ", tpc=" << mf->get_tpc() << hex << ", PC=0x" << PC << ", type=" << mem_access_type_str(accessType) << endl;
        fflush(stderr);
    }

    return victim_idx;
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
int tag_array::Get_LRU_Victim( unsigned setIndex, unsigned used_idx )
{
    // Get pointer to replacement state of current set
    LINE_REPLACEMENT_STATE *replSet = repl[ setIndex ];

    int   lruWay   = -1;
    unsigned maxLRUstackposition = 0;

    // Search for victim whose stack position is m_assoc-1
    for(unsigned way=0; way<m_assoc; way++) 
    {
        unsigned idx = setIndex * m_assoc + way;
        if( replSet[way].LRUstackposition >= maxLRUstackposition && m_lines[idx].m_status != RESERVED && m_lines[idx].m_status != INVALID && m_lines[idx].m_status != UNUSED && way != used_idx )
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

BypassMonitor::BypassMonitor(cache_config cfg)
{
    char c_repl;
    int ntok = sscanf(cfg.m_config_repl_string,"%c:%u:%u,%u:%u:%u,%u:%u:%c,%u",
                      &c_repl, &m_rrpv_nbit, &NumThreadsPerCache, &numsets, &assoc, &m_update_pr,
                      &m_bd_addr_len, &m_bd_nbit, &m_bd_idx, &m_bm_partial_tag_nbit);

    if ( ntok != 10 )
    {
        printf("GPGPU-Sim uArch: cache replacement configuration parsing error (%s)\n", cfg.m_config_repl_string);
        abort();
    }

    switch (c_repl) {
    case 'L': m_base_repl = BASE_REPL_LRU; break;
    case 'N': m_base_repl = BASE_REPL_NRU; break;
    case 'S': m_base_repl = BASE_REPL_SRRIP; break;
    case 'P': m_base_repl = BASE_REPL_PDP; break;
    case 'R': m_base_repl = BASE_REPL_RANDOM; break;
    default: abort();
    }

    m_prefetch_aware = cfg.m_enable_prefetch;
    m_cache_config = cfg;
    if(m_bm_partial_tag_nbit > 0 && cfg.m_set_index_function == FERMI_HASH_SET_FUNCTION)
        m_bm_partial_tag_nbit += indexShift;

    if(m_prefetch_aware)
        m_bd_max_addr = (1 << (m_bd_addr_len + 1));
    else
        m_bd_max_addr = (1 << m_bd_addr_len);

    linesize = cfg.get_line_sz();
    lineShift  = LOGB2( linesize );
    indexShift = LOGB2( numsets );
    indexMask  = (1 << indexShift) - 1;

    misses = 0;
    BypassDecider = new int* [NumThreadsPerCache];
    assert(BypassDecider);
    BD_saturate_counter = new unsigned[NumThreadsPerCache];
    assert(BD_saturate_counter);
    access_num = new unsigned[NumThreadsPerCache];
    assert(access_num);
    hititag_num = new unsigned[NumThreadsPerCache];
    assert(hititag_num);
    hitvtag_num = new unsigned[NumThreadsPerCache];
    assert(hitvtag_num);
    evict_num = new unsigned[NumThreadsPerCache];
    assert(evict_num);
    evicted_num = new unsigned[NumThreadsPerCache];
    assert(evicted_num);
    vichititag_num = new unsigned[NumThreadsPerCache];
    assert(vichititag_num);

    for(unsigned i = 0; i < NumThreadsPerCache; i++)
    {
        BypassDecider[i] = new int[m_bd_max_addr];
        assert(BypassDecider[i]);
        for(unsigned j = 0; j < m_bd_max_addr; j++)
            BypassDecider[i][j] = -1;
        //for(unsigned j = (1 << m_bd_addr_len); j < m_bd_max_addr; j++)
        //    BypassDecider[i][j] = 0;
        BD_saturate_counter[i] = 0;

        access_num[i] = 0;
        hititag_num[i] = 0;
        hitvtag_num[i] = 0;
        vichititag_num[i] = 0;
        evict_num[i] = 0;
        evicted_num[i] = 0;
    }

    // Create the state for sets, then create the state for the ways
    entries = new BM_entry* [ numsets ];

    // ensure that we were able to create replacement state
    assert(entries);

    // Create the state for the sets
    for(unsigned setIndex=0; setIndex<numsets; setIndex++) 
    {
        entries[ setIndex ]  = new BM_entry[ assoc ];

        for(unsigned way=0; way<assoc; way++) 
        {
            // initialize stack position (for true LRU)
            entries[ setIndex ][ way ].LRUstackposition = way;
            entries[ setIndex ][ way ].valid = false;
        }
    }
}

int BypassMonitor::Get_Victim( unsigned tid, unsigned setIndex )
{
    // Get pointer to replacement state of current set
    BM_entry *replSet = entries[ setIndex ];

    int   lruWay   = 0;

    // Search for invalid victim
    for(unsigned way=0; way<assoc; way++) 
    {
        if( replSet[way].valid == false ) 
        {
            UpdateLRU(setIndex, way);
            return way;
        }
    }

    if(misses % m_update_pr != 0)
        return -1;

    // Search for victim whose stack position is assoc-1
    for(unsigned way=0; way<assoc; way++) 
    {
        if( replSet[way].LRUstackposition == (assoc-1) ) 
        {
            lruWay = way;
            break;
        }
    }

    evict_num[tid]++;
    evicted_num[replSet[lruWay].Itid]++;
    UpdateLRU(setIndex, lruWay);

    // return lru way
    return lruWay;
}

void BypassMonitor::Fill(int set, int way, unsigned tid, address_type pc, new_addr_type addr, new_addr_type vaddr, int sign, mem_access_type type)
{
    entries[set][way].valid = true;
    entries[set][way].Itid = tid;
    entries[set][way].Itype = type;
    entries[set][way].Ipc = pc;
    entries[set][way].Isign = sign;
    entries[set][way].Itag = GetTag(addr);
    entries[set][way].Vtag = GetTag(vaddr);
}

int BypassMonitor::LookUp(unsigned tid, new_addr_type addr)
{
    int   set = 0;
    int hititag = 0;
    int hitvtag = 0;

    set = GetSetIndex(addr);

    access_num[tid]++;
    // Get pointer to current set
    BM_entry *currSet = entries[ set ];

    // Find Tag
    for(unsigned way=0; way<assoc; way++) 
    {
        if( currSet[way].valid )
        {
            if(currSet[way].Itag == GetTag(addr))
            {
                UpdateBD(tid, currSet[way].Ipc, addr, GetAddr(set, currSet[way].Vtag), currSet[way].Isign, false, currSet[way].LRUstackposition, currSet[way].Itype);
                currSet[way].valid = false;
                hititag_num[tid]++;
                hititag++;
                if(m_bm_partial_tag_nbit == 0 && tid != currSet[way].Itid)
                    cerr<<"Error: tid is not same."<<endl;
            }
            if(currSet[way].Vtag == GetTag(addr))
            {
                UpdateBD(currSet[way].Itid, currSet[way].Ipc, GetAddr(set, currSet[way].Itag), currSet[way].Isign, addr, true, currSet[way].LRUstackposition, currSet[way].Itype);
                currSet[way].valid = false;
                hitvtag_num[tid]++;
                hitvtag++;
            }
        }
    }
    return hititag - hitvtag;
}

void BypassMonitor::LookUpVictim(unsigned tid, new_addr_type addr)
{
    int   set = 0;

    set = GetSetIndex(addr);

    access_num[tid]++;
    // Get pointer to current set
    BM_entry *currSet = entries[ set ];

    // Find Tag
    for(unsigned way=0; way<assoc; way++) 
    {
        if( currSet[way].valid )
        {
            if(currSet[way].Itag == GetTag(addr))
            {
                UpdateBD(tid, currSet[way].Ipc, addr, GetAddr(set, currSet[way].Vtag), currSet[way].Isign, true, currSet[way].LRUstackposition, currSet[way].Itype);
                vichititag_num[tid]++;
                currSet[way].valid = false;
            }
        }
    }
}

void BypassMonitor::UpdateLRU( unsigned setIndex, int updateWayID )
{
    // Determine current LRU stack position
    unsigned currLRUstackposition = entries[ setIndex ][ updateWayID ].LRUstackposition;

    // Update the stack position of all lines before the current line
    // Update implies incremeting their stack positions by one
    for(unsigned way=0; way<assoc; way++) 
    {
        if( entries[setIndex][way].LRUstackposition < currLRUstackposition ) 
        {
            entries[setIndex][way].LRUstackposition++;
        }
    }

    // Set the LRU stack position of new line to be zero
    entries[ setIndex ][ updateWayID ].LRUstackposition = 0;
}

void BypassMonitor::UpdateBD(unsigned tid, new_addr_type PC, new_addr_type addr, new_addr_type vaddr, int sign, bool direction, unsigned weight, unsigned accessType)
{
    // Generate BD update address
    new_addr_type bd_addr = GetBDaddr(tid, PC, addr, vaddr, sign, accessType);

    if(direction == true)  // Add
    {
        BypassDecider[tid][bd_addr] = SaturateAdd(BypassDecider[tid][bd_addr], 1, BD_MAX);
        if(BypassDecider[tid][bd_addr] == BD_MAX)
            BD_saturate_counter[tid]++;
    }
    else  // Sub
    {
        BypassDecider[tid][bd_addr] = SaturateSub(BypassDecider[tid][bd_addr], 1, BD_MAX);
        if(BypassDecider[tid][bd_addr] == -(BD_MAX + 1))
            BD_saturate_counter[tid]++;
    }
}

bool BypassMonitor::IsBypass(unsigned tid, new_addr_type PC, new_addr_type addr, new_addr_type vaddr, int sign, unsigned accessType)
{
    // Generate BD update address
    new_addr_type bd_addr = GetBDaddr(tid, PC, addr, vaddr, sign, accessType);

    if(BypassDecider[tid][bd_addr] >= 0)
        return true;
    else
        return false;
}

new_addr_type BypassMonitor::GetBDaddr(unsigned &tid, new_addr_type PC, new_addr_type addr, new_addr_type vaddr, int sign, unsigned accessType)
{
    new_addr_type bd_addr;
    if(m_bm_partial_tag_nbit > 0)
        addr &= (1 << (BM_PARTIAL_TAG_BIT + indexShift + lineShift)) - 1;

    switch (m_bd_idx) {
    case 'p':
        bd_addr = (PC >> BD_PC_SHIFT) & BD_ADDR_MASK;
        break;
    case 'm':
        bd_addr = ((addr >> (lineShift + BD_ADDR_SHIFT)) & BD_ADDR_MASK);
        for(unsigned i = 0; i < m_bd_addr_len; i++)
            bd_addr ^= ((((PC >> BD_PC_SHIFT) & BD_ADDR_MASK) >> i) & 1) << (m_bd_addr_len - 1 - i);
        break;
    default:
        bd_addr = ((addr >> (lineShift + BD_ADDR_SHIFT)) & BD_ADDR_MASK);
    }

    if(m_prefetch_aware && 
       (accessType == L1_PREF_ACC_R || accessType == L2_PREF_ACC_R)) {
        bd_addr += (1 << m_bd_addr_len) + (sign << (m_bd_addr_len - 2)); // FIXME
        bd_addr %= m_bd_max_addr;
    }

    if(m_bd_idx == 'p')
        tid = 0;

    assert(bd_addr < m_bd_max_addr);
    return bd_addr;
}

new_addr_type BypassMonitor::GetTag( new_addr_type addr )
{
    if(m_bm_partial_tag_nbit > 0) {
        if(m_cache_config.m_set_index_function == FERMI_HASH_SET_FUNCTION)
            return ((addr >> lineShift) & BM_PARTIAL_TAG_MASK);
        else
            return (((addr >> lineShift) >> indexShift) & BM_PARTIAL_TAG_MASK);
    } else {
        if(m_cache_config.m_set_index_function == FERMI_HASH_SET_FUNCTION)
            return (addr >> lineShift);
        else
            return ((addr >> lineShift) >> indexShift);
    }
}

int BypassMonitor::SaturateAdd(int x, int inc, int max)
{
    x += inc;

    if(x < max)
        return x;
    else
        return max;
}

int BypassMonitor::SaturateSub(int x, int inc, int max)
{
    x -= inc;

    if(x > -(max + 1))
        return x;
    else
        return -(max + 1);
}

void BypassMonitor::Print(ostream &out)
{
    unsigned i, j;

    out<<"=========== Bypass Monitor ==========="<<endl;

    for(i = 0; i < numsets; i++)
    {
        out<<"Set "<<i<<": "<<endl;
        for(j = 0; j < assoc; j++)
        {
            if(entries[i][j].valid)
            {
                out<<"Position: "<<entries[i][j].LRUstackposition<<endl;
                out<<"Itid    : "<<entries[i][j].Itid<<endl;
                out<<"IAddr   : "<<GetAddr(i, entries[i][j].Itag)<<endl;
                out<<"VAddr   : "<<GetAddr(i, entries[i][j].Vtag)<<endl;
            }
        }
    }
}

void BypassMonitor::PrintStats(ostream &out)
{
    int allaccesses;
    int allhits;

    for(unsigned i = 0; i < NumThreadsPerCache; i++)
    {
        out<<"\tBM stat "<<i<<": ";
        out<<"access_num="          <<access_num[i]<<", ";
        out<<"hit_itag_num="        <<hititag_num[i]<<", ";
        out<<"hit_vtag_num="        <<hitvtag_num[i]<<", ";
        out<<"vic_hit_itag_num="    <<vichititag_num[i]<<", ";
        out<<"evict_num="           <<evict_num[i]<<", ";
        out<<"evicted_num="         <<evicted_num[i]<<", ";
        allhits = hititag_num[i] + hitvtag_num[i] + vichititag_num[i];
        allaccesses = allhits + evicted_num[i] + numsets * assoc / NumThreadsPerCache;
        out<<"hit_rate="            <<(double)allhits / (double)allaccesses * 100.0<<", ";
        out<<"saturate_num="        <<BD_saturate_counter[i]<<endl;
    }
}
