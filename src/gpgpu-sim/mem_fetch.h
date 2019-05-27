// Copyright (c) 2009-2011, Tor M. Aamodt
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

#ifndef MEM_FETCH_H
#define MEM_FETCH_H

#include "addrdec.h"
#include "../abstract_hardware_model.h"
#include <bitset>

enum cache_request_status {
    HIT = 0,
    HIT_RESERVED,
    MISS,
    MISS_PARTIAL, // lld: sector cache
    RESERVATION_FAIL, 
    NUM_CACHE_REQUEST_STATUS
};

enum mf_type {
   READ_REQUEST = 0,
   WRITE_REQUEST,
   READ_REPLY, // send to shader
   WRITE_ACK
};

#define MF_TUP_BEGIN(X) enum X {
#define MF_TUP(X) X
#define MF_TUP_END(X) };
#include "mem_fetch_status.tup"
#undef MF_TUP_BEGIN
#undef MF_TUP
#undef MF_TUP_END

class mem_fetch {
public:
    mem_fetch( const mem_access_t &access, 
               const warp_inst_t *inst,
               unsigned ctrl_size, 
               unsigned wid,
               unsigned sid, 
               unsigned tpc, 
               const class memory_config *config );
   ~mem_fetch();

   void set_status( enum mem_fetch_status status, unsigned long long cycle );
   void set_reply() 
   { 
       assert( m_access.get_type() != L1_WRBK_ACC && m_access.get_type() != L2_WRBK_ACC );
       if( m_type==READ_REQUEST ) {
           assert( !get_is_write() );
           m_type = READ_REPLY;
       } else if( m_type == WRITE_REQUEST ) {
           assert( get_is_write() );
           m_type = WRITE_ACK;
       }
   }
   void do_atomic();

   void print( FILE *fp, bool print_inst = true ) const;

   const addrdec_t &get_tlx_addr() const { return m_raw_addr; }
   unsigned get_data_size() const { return m_data_size; }
   void     set_data_size( unsigned size ) { m_data_size=size; }
   unsigned get_ctrl_size() const { return m_ctrl_size; }
   unsigned size() const { return m_data_size+m_ctrl_size; }
   bool is_write() {return m_access.is_write();}
   void set_addr(new_addr_type addr) { m_access.set_addr(addr); }
   new_addr_type get_addr() const { return m_access.get_addr(); }
   new_addr_type get_partition_addr() const { return m_partition_addr; }
   unsigned get_sub_partition_id() const { return m_raw_addr.sub_partition; }
   bool     get_is_write() const { return m_access.is_write(); }
   unsigned get_request_uid() const { return m_request_uid; }
   unsigned get_sid() const { return m_sid; }
   unsigned get_tpc() const { return m_tpc; }
   unsigned get_wid() const { return m_wid; }
   bool istexture() const;
   bool isconst() const;
   enum mf_type get_type() const { return m_type; }
   bool isatomic() const;

   void set_return_timestamp( unsigned t ) { m_timestamp2=t; }
   void set_icnt_receive_time( unsigned t ) { m_icnt_receive_time=t; }
   unsigned get_timestamp() const { return m_timestamp; }
   unsigned get_return_timestamp() const { return m_timestamp2; }
   unsigned get_icnt_receive_time() const { return m_icnt_receive_time; }

   // lld: for memory latency calculation
   void set_issue(bool l2) { if(l2) m_l2_issue=true; else m_issue=true; }
   bool get_issue(bool l2) const { if(l2) return m_l2_issue; else return m_issue; }
   void set_issue_time( bool l2, unsigned t ) { if(l2) m_l2_issue_time=t; else m_l1_issue_time=t; }
   void set_ret_time( bool l2, unsigned t ) { if(l2) m_l2_ret_time=t; else m_l1_ret_time=t; }
   unsigned get_issue_time(bool l2) const { if(l2) return m_l2_issue_time; else return m_l1_issue_time; }
   unsigned get_ret_time(bool l2) const { if(l2) return m_l2_ret_time; else return m_l1_ret_time; }

   // lld: additional information
   void set_access_status( bool l2, cache_request_status status ) { if(l2) m_l2_status=status; else m_l1_status=status; }
   cache_request_status get_status( bool l2 ) { if(l2) return m_l2_status; else return m_l1_status; }
   std::bitset<MAX_CACHE_CHUNK_NUM> get_original_sector_mask( bool l2 ) const { if(l2) return m_l2_original_sector_mask; else return m_original_sector_mask; }
   void set_original_sector_mask(bool l2) { assert((!l2 && !m_l1_combination) || (l2 && !m_l2_combination));
                                            if(!l2) m_original_sector_mask = m_sector_mask; m_l2_original_sector_mask = m_l2_sector_mask; }
   std::bitset<MAX_CACHE_CHUNK_NUM> get_sector_mask( bool l2 ) const { if(l2) return m_l2_sector_mask; else return m_sector_mask; }
   void set_sector_mask(bool l2, unsigned i, bool value) { if(!l2) { if(value) m_sector_mask.set(i); else m_sector_mask.reset(i); }
                                                           if(value) m_l2_sector_mask.set(i); else m_l2_sector_mask.reset(i); }
   std::bitset<MAX_CACHE_CHUNK_NUM> get_alloc_sector_mask( bool l2 ) const { if(l2) return m_l2_alloc_sector_mask; else return m_alloc_sector_mask; }
   void set_alloc_sector_mask(bool l2, unsigned i, bool value) { if(!l2) { if(value) m_alloc_sector_mask.set(i); else m_alloc_sector_mask.reset(i); }
                                                           if(value) m_l2_alloc_sector_mask.set(i); else m_l2_alloc_sector_mask.reset(i); }
   bool get_alloc_on_fill(bool l2) { if(l2) return m_l2_alloc_on_fill; else return m_l1_alloc_on_fill; }
   void set_alloc_on_fill(bool l2) { if(l2) m_l2_alloc_on_fill = true; else m_l1_alloc_on_fill = true; }
   bool get_prefetch(bool l2) { if(l2) return m_l2_prefetch; else return m_l1_prefetch; }
   int get_sign() { return m_sign; }
   void set_prefetch_info(bool l2, address_type pc, int sign) { if(l2) m_l2_prefetch=true; else m_l1_prefetch=true;
                                                                          m_my_pc = pc; m_sign = sign; }
   void set_done(bool l2, bool value) { if(l2) m_l2_done=value; else m_l1_done=value; }
   bool get_done(bool l2) { if(l2) return m_l2_done; else return m_l1_done; }
   void set_write_sent(bool l2) { if(l2) m_l2_write_sent=true; else m_l1_write_sent=true; }
   bool get_write_sent(bool l2) { if(l2) return m_l2_write_sent; else return m_l1_write_sent; }
   bool get_first_touch(bool l2)
   {
       if(l2 && m_l2_first_touch) {
           m_l2_first_touch = false;
           return true;
       } else if(!l2 && m_l1_first_touch) {
           m_l1_first_touch = false;
           return true;
       }
       return false;
   }

   std::list<mem_fetch*> m_merged_requests; // lld: adaptive granularity

   enum mem_access_type get_access_type() const { return m_access.get_type(); }
   const active_mask_t& get_access_warp_mask() const { return m_access.get_warp_mask(); }
   mem_access_byte_mask_t get_access_byte_mask() const { return m_access.get_byte_mask(); }
   void set_access_byte_mask(unsigned i, bool value) { m_access.set_byte_mask(i, value); } // lld: sector cache
   unsigned get_combination(bool l2) { if(l2) return m_l2_combination; else return m_l1_combination; }
   void inc_combination(bool l2) { if(l2) m_l2_combination++; else m_l1_combination++; }

   address_type get_pc() const { if(get_access_type() == L1_PREF_ACC_R || get_access_type() == L2_PREF_ACC_R) return m_my_pc; // lld: for prefetch
                                 else return m_inst.empty()?-1:m_inst.pc; }
   const warp_inst_t &get_inst() { return m_inst; }
   enum mem_fetch_status get_status() const { return m_status; }

   const memory_config *get_mem_config(){return m_mem_config;}

   unsigned get_num_flits(bool simt_to_mem);
private:
   // request source information
   unsigned m_request_uid;
   unsigned m_sid;
   unsigned m_tpc;
   unsigned m_wid;

   // where is this request now?
   enum mem_fetch_status m_status;
   unsigned long long m_status_change;

   // request type, address, size, mask
   mem_access_t m_access;
   unsigned m_data_size; // how much data is being written
   unsigned m_ctrl_size; // how big would all this meta data be in hardware (does not necessarily match actual size of mem_fetch)
   new_addr_type m_partition_addr; // linear physical address *within* dram partition (partition bank select bits squeezed out)
   addrdec_t m_raw_addr; // raw physical address (i.e., decoded DRAM chip-row-bank-column address)
   enum mf_type m_type;

   // statistics
   unsigned m_timestamp;  // set to gpu_sim_cycle+gpu_tot_sim_cycle at struct creation
   unsigned m_timestamp2; // set to gpu_sim_cycle+gpu_tot_sim_cycle when pushed onto icnt to shader; only used for reads
   unsigned m_icnt_receive_time; // set to gpu_sim_cycle + interconnect_latency when fixed icnt latency mode is enabled

   // lld
   address_type m_my_pc;
   int m_sign;
   bool m_issue;
   bool m_l2_issue;
   unsigned m_l1_issue_time;
   unsigned m_l2_issue_time;
   unsigned m_l1_ret_time;
   unsigned m_l2_ret_time;
   cache_request_status m_l1_status;
   cache_request_status m_l2_status;
   std::bitset<MAX_CACHE_CHUNK_NUM> m_original_sector_mask; // lld: sector cache
   std::bitset<MAX_CACHE_CHUNK_NUM> m_sector_mask; // lld: sector cache
   std::bitset<MAX_CACHE_CHUNK_NUM> m_alloc_sector_mask; // lld: adaptive line size
   std::bitset<MAX_CACHE_CHUNK_NUM> m_l2_original_sector_mask; // lld: sector cache
   std::bitset<MAX_CACHE_CHUNK_NUM> m_l2_sector_mask; // lld: sector cache
   std::bitset<MAX_CACHE_CHUNK_NUM> m_l2_alloc_sector_mask; // lld: adaptive line size
   unsigned m_l1_combination;
   unsigned m_l2_combination;
   bool m_l1_alloc_on_fill;
   bool m_l2_alloc_on_fill;
   bool m_l1_prefetch;
   bool m_l2_prefetch;
   bool m_l1_done;
   bool m_l2_done;
   bool m_l1_write_sent;
   bool m_l2_write_sent;
   bool m_l1_first_touch;
   bool m_l2_first_touch;

   // requesting instruction (put last so mem_fetch prints nicer in gdb)
   warp_inst_t m_inst;

   static unsigned sm_next_mf_request_uid;

   const class memory_config *m_mem_config;
   unsigned icnt_flit_size;
};

#endif
