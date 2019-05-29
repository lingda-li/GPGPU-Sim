#include "../cuda-sim/cuda-sim.h"
#include "gpu-cache.h"
#include <iostream>
#include <sstream>
#include <assert.h>

using namespace std;

// decide which parts should be allocated
void tag_array::decide_allocation(mem_fetch *mf, new_addr_type addr)
{
    unsigned tid = mf->get_wid() % BM->NumThreadsPerCache; //FIXME: how to generate tid?
    address_type PC = mf->get_pc();
    mem_access_type accessType = mf->get_access_type();

    // decide whether to allocate for demand data
    //if(BM->IsBypass(tid, PC, addr, victim_addr, mf->get_sign(), accessType))
    if(BM->IsBypass(tid, PC, addr, 0, 0, accessType))
    {
        // clean allocated sector mask
        for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++)
            mf->set_alloc_sector_mask(m_core_id<0, i, false);
        assert(!mf->get_alloc_sector_mask(m_core_id<0).count());
    }
}

void tag_array::decide_prefetch(mem_fetch *mf, new_addr_type addr)
{
    unsigned tid = mf->get_wid() % BM->NumThreadsPerCache; //FIXME: how to generate tid?
    address_type PC = mf->get_pc();
    mem_access_type accessType = mf->get_access_type();

    // decide whether to fetch and allocate for extra data
    for(int i = 0; i < (int)(m_config.get_line_sz()/CACHE_CHUNK_SIZE); i++)
        if(!mf->get_original_sector_mask(m_core_id<0).test(i)) {
            int j;
            bool pos_res = true, neg_res = true;
            // find the nearest front demand chunk
            for(j = i-1; j >= 0; j--)
                if(mf->get_original_sector_mask(m_core_id<0).test(j))
                    break;
            if(j >= 0) {
                pos_res = BM->IsBypass(tid, PC, addr, 0, i - j, accessType);
                if(pos_res)
                    continue;
            }
            // find the nearest latter demand chunk
            for(j = i+1; j < (int)(m_config.get_line_sz()/CACHE_CHUNK_SIZE); j++)
                if(mf->get_original_sector_mask(m_core_id<0).test(j))
                    break;
            if(j < (int)(m_config.get_line_sz()/CACHE_CHUNK_SIZE)) {
                neg_res = BM->IsBypass(tid, PC, addr, 0, i - j, accessType);
                if(neg_res)
                    continue;
            }

            assert(!pos_res || !neg_res);
            mf->set_sector_mask(m_core_id<0, i, true);
            mf->set_alloc_sector_mask(m_core_id<0, i, true);
        }
}

unsigned tag_array::get_victim(mem_fetch *mf, new_addr_type addr)
{
    assert(m_config.m_replacement_policy == ADAP_GRAN);

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
    if(accessType == L1_WRBK_ACC)
        return (unsigned)-1;
    BM->misses++;

    bool is_bypass = true;
    bool is_neutral = true;
    for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++) {
        if(mf->get_original_sector_mask(m_core_id<0).test(i)) {
            if(mf->get_alloc_sector_mask(m_core_id<0).test(i)) {
                assert(!is_bypass || is_neutral);
                is_neutral = false;
                is_bypass = false;
            } else {
                assert(is_bypass || is_neutral);
                is_neutral = false;
                is_bypass = true;
            }
        } else if(mf->get_alloc_sector_mask(m_core_id<0).test(i)) {
            m_issued_prefetch++;
        }
    }
    assert(!is_neutral);
    if(is_bypass)
        m_bypasses++;
    else
        m_nonbypasses++;

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
    BM->LookUpVictim(tid, &m_lines[victim_idx]);

    // Update Bypass Monitor
    int BM_victim_set = BM->GetSetIndex(addr);
    int BM_victim_way = BM->Get_Victim(tid, BM_victim_set);
    if(BM_victim_way >= 0)
        BM->Fill(BM_victim_set, BM_victim_way, mf, tid, PC, addr, &m_lines[victim_idx], mf->get_sign(), accessType, m_core_id<0);

    if(is_bypass)
        victim_idx = (unsigned)-1;
    /*
    // Make bypassing decision
    //if(BM->IsBypass(tid, PC, addr, victim_addr, mf->get_sign(), accessType))
    if(BM->IsBypass(tid, PC, addr, victim_addr, 0, accessType))
    {
        victim_idx = (unsigned)-1;
        m_bypasses++;
    }
    else
        m_nonbypasses++;
       */

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
    if(accessType == L1_WRBK_ACC)
        return;
    if(!is_real_access)
        return;

    bool hit = (status == HIT) || (status == MISS_PARTIAL);

    if(hit) // there should be no L2 writeback
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
    else
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
    if(!is_real_access)
        return;

    BM->LookUp(mf, tid, addr, m_core_id<0); // Check Bypass Monitor

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
    if(accessType == L1_WRBK_ACC)
        return (unsigned)-1;
    BM->misses++;

    bool is_bypass = true;
    bool is_neutral = true;
    for(unsigned i = 0; i < m_config.get_line_sz()/CACHE_CHUNK_SIZE; i++) {
        if(mf->get_original_sector_mask(m_core_id<0).test(i)) {
            if(mf->get_alloc_sector_mask(m_core_id<0).test(i)) {
                assert(!is_bypass || is_neutral);
                is_neutral = false;
                is_bypass = false;
            } else {
                assert(is_bypass || is_neutral);
                is_neutral = false;
                is_bypass = true;
            }
        } else if(mf->get_alloc_sector_mask(m_core_id<0).test(i)) {
            m_issued_prefetch++;
        }
    }
    assert(!is_neutral);
    if(is_bypass)
        m_bypasses++;
    else
        m_nonbypasses++;

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
    BM->LookUpVictim(tid, &m_lines[victim_idx]);

    // Update Bypass Monitor
    int BM_victim_set = BM->GetSetIndex(addr);
    int BM_victim_way = BM->Get_Victim(tid, BM_victim_set);
    if(BM_victim_way >= 0)
        BM->Fill(BM_victim_set, BM_victim_way, mf, tid, PC, addr, &m_lines[victim_idx], mf->get_sign(), accessType, m_core_id<0);

    if(is_bypass)
        victim_idx = (unsigned)-1;
    /*
    // Make bypassing decision
    //if(BM->IsBypass(tid, PC, addr, victim_addr, mf->get_sign(), accessType))
    if(BM->IsBypass(tid, PC, addr, victim_addr, 0, accessType))
    {
        victim_idx = (unsigned)-1;
        m_bypasses++;
    }
    else
        m_nonbypasses++;
       */

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
    BM->LookUpVictim(tid, &m_lines[victim_idx]);

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

    if(cfg.m_replacement_policy == ADAP_GRAN)
        m_bd_max_addr = (1 << m_bd_addr_len) * (2*cfg.get_line_sz()/CACHE_CHUNK_SIZE-1);
    else if(m_prefetch_aware)
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
        for(unsigned j = (1 << m_bd_addr_len); j < m_bd_max_addr; j++)
            BypassDecider[i][j] = 0;
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

void BypassMonitor::Fill(int set, int way, mem_fetch *mf, unsigned tid, address_type pc, new_addr_type addr, cache_block_t *vic_blk, int sign, mem_access_type type, bool l2)
{
    entries[set][way].valid = true;
    entries[set][way].Itid = tid;
    entries[set][way].Itype = type;
    entries[set][way].Ipc = pc;
    entries[set][way].Isign = sign;
    entries[set][way].Itag = GetTag(addr);
    entries[set][way].Vtag = GetTag(vic_blk->m_block_addr);
    entries[set][way].i_sector_mask = mf->get_original_sector_mask(l2);
    entries[set][way].v_sector_mask = vic_blk->m_sector_mask;
}

int BypassMonitor::LookUp(mem_fetch *mf, unsigned tid, new_addr_type addr, bool l2)
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
                // demand parts
                bitset<MAX_CACHE_CHUNK_NUM> sector_mask = mf->get_original_sector_mask(l2) & currSet[way].i_sector_mask;
                if(sector_mask.count() > 0) {
                    UpdateBD(tid, currSet[way].Ipc, addr, GetAddr(set, currSet[way].Vtag), 0, false, currSet[way].LRUstackposition, currSet[way].Itype);
                    currSet[way].valid = false;
                    hititag_num[tid]++;
                    hititag++;
                }

                // other parts
                for(int i = 0; i < (int)(linesize/CACHE_CHUNK_SIZE); i++) {
                    if(mf->get_original_sector_mask(l2).test(i) && !currSet[way].i_sector_mask.test(i)) { // prefetch will benefit
                        // find the nearest front demand chunk
                        int j;
                        for(j = i-1; j >= 0; j--)
                            if(currSet[way].i_sector_mask.test(j))
                                break;
                        if(j >= 0) {
                            UpdateBD(tid, currSet[way].Ipc, addr, GetAddr(set, currSet[way].Vtag), i - j, false, currSet[way].LRUstackposition, currSet[way].Itype);
                            currSet[way].valid = false;
                            hititag_num[tid]++;
                            hititag++;
                        }
                        // find the nearest latter demand chunk
                        for(j = i+1; j < (int)(linesize/CACHE_CHUNK_SIZE); j++)
                            if(currSet[way].i_sector_mask.test(j))
                                break;
                        if(j < (int)(linesize/CACHE_CHUNK_SIZE)) {
                            UpdateBD(tid, currSet[way].Ipc, addr, GetAddr(set, currSet[way].Vtag), i - j, false, currSet[way].LRUstackposition, currSet[way].Itype);
                            currSet[way].valid = false;
                            hititag_num[tid]++;
                            hititag++;
                        }
                    }
                }
                if(m_bm_partial_tag_nbit == 0 && tid != currSet[way].Itid)
                    cerr<<"Error: tid is not same."<<endl;
            }
            if(currSet[way].Vtag == GetTag(addr))
            {
                bitset<MAX_CACHE_CHUNK_NUM> sector_mask = mf->get_original_sector_mask(l2) & currSet[way].v_sector_mask;
                if(sector_mask.count() > 0) {
                    // demand parts
                    UpdateBD(currSet[way].Itid, currSet[way].Ipc, GetAddr(set, currSet[way].Itag), addr, 0, true, currSet[way].LRUstackposition, currSet[way].Itype);
                    currSet[way].valid = false;
                    hitvtag_num[tid]++;
                    hitvtag++;

                    // other parts
                    for(int i = 0; i < (int)(linesize/CACHE_CHUNK_SIZE); i++) {
                        if(!currSet[way].i_sector_mask.test(i)) { // prefetch will suck
                            // find the nearest front demand chunk
                            int j;
                            for(j = i-1; j >= 0; j--)
                                if(currSet[way].i_sector_mask.test(j))
                                    break;
                            if(j >= 0) {
                                UpdateBD(currSet[way].Itid, currSet[way].Ipc, GetAddr(set, currSet[way].Itag), addr, i - j, true, currSet[way].LRUstackposition, currSet[way].Itype);
                                currSet[way].valid = false;
                                hitvtag_num[tid]++;
                                hitvtag++;
                            }
                            // find the nearest latter demand chunk
                            for(j = i+1; j < (int)(linesize/CACHE_CHUNK_SIZE); j++)
                                if(currSet[way].i_sector_mask.test(j))
                                    break;
                            if(j < (int)(linesize/CACHE_CHUNK_SIZE)) {
                                UpdateBD(currSet[way].Itid, currSet[way].Ipc, GetAddr(set, currSet[way].Itag), addr, i - j, true, currSet[way].LRUstackposition, currSet[way].Itype);
                                currSet[way].valid = false;
                                hitvtag_num[tid]++;
                                hitvtag++;
                            }
                        }
                    }
                } else { // update victim's sector mask
                    for(unsigned i = 0; i < linesize/CACHE_CHUNK_SIZE; i++)
                        if(mf->get_original_sector_mask(l2).test(i))
                            currSet[way].v_sector_mask.test(i);
                }
            }
        }
    }
    return hititag - hitvtag;
}

void BypassMonitor::LookUpVictim(unsigned tid, cache_block_t *vic_blk)
{
    int   set = 0;
    new_addr_type addr = vic_blk->m_block_addr;

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
                bitset<MAX_CACHE_CHUNK_NUM> sector_mask = vic_blk->m_sector_mask & currSet[way].i_sector_mask;
                if(sector_mask.count() > 0) {
                    // demand parts
                    UpdateBD(tid, currSet[way].Ipc, addr, GetAddr(set, currSet[way].Vtag), 0, true, currSet[way].LRUstackposition, currSet[way].Itype);
                    currSet[way].valid = false;
                    vichititag_num[tid]++;

                    // other parts
                    for(int i = 0; i < (int)(linesize/CACHE_CHUNK_SIZE); i++) {
                        //if(!currSet[way].i_sector_mask.test(i)) { // prefetch will suck
                        if(!currSet[way].i_sector_mask.test(i) && vic_blk->m_sector_mask.test(i)) { // prefetch will suck
                            // find the nearest front demand chunk
                            int j;
                            for(j = i-1; j >= 0; j--)
                                if(currSet[way].i_sector_mask.test(j))
                                    break;
                            if(j >= 0) {
                                UpdateBD(tid, currSet[way].Ipc, addr, GetAddr(set, currSet[way].Vtag), i - j, true, currSet[way].LRUstackposition, currSet[way].Itype);
                                currSet[way].valid = false;
                                vichititag_num[tid]++;
                            }
                            // find the nearest latter demand chunk
                            for(j = i+1; j < (int)(linesize/CACHE_CHUNK_SIZE); j++)
                                if(currSet[way].i_sector_mask.test(j))
                                    break;
                            if(j < (int)(linesize/CACHE_CHUNK_SIZE)) {
                                UpdateBD(tid, currSet[way].Ipc, addr, GetAddr(set, currSet[way].Vtag), i - j, true, currSet[way].LRUstackposition, currSet[way].Itype);
                                currSet[way].valid = false;
                                vichititag_num[tid]++;
                            }
                        }
                    }
                }
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

    //if(m_prefetch_aware && 
    //   (accessType == L1_PREF_ACC_R || accessType == L2_PREF_ACC_R)) {
    //    bd_addr += (1 << m_bd_addr_len) + (sign << (m_bd_addr_len - 2)); // FIXME
    //    bd_addr %= m_bd_max_addr;
    //}
    assert(sign > -(int)(linesize/CACHE_CHUNK_SIZE) && sign < (int)(linesize/CACHE_CHUNK_SIZE));
    if(sign > 0)
        bd_addr += (sign << m_bd_addr_len);
    else if(sign < 0)
        bd_addr += ((sign + (int)(2*linesize/CACHE_CHUNK_SIZE) - 1) << m_bd_addr_len);

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
