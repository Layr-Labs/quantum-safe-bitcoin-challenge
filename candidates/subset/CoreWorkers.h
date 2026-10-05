// SPDX-License-Identifier: GPL-3.0-only
// A conservative, topology-derived steady-state worker plan. Host-only.
#pragma once
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <vector>
namespace qsb_core {
inline bool parse_list(const std::string &s, int limit, std::vector<int> &out) {
    out.clear(); std::size_t p=0;
    auto number=[&](int &v) {
        if(p==s.size() || s[p]<'0' || s[p]>'9') return false;
        v=0;
        while(p<s.size() && s[p]>='0' && s[p]<='9') {
            const int digit=s[p++]-'0';
            if(v>(limit-1-digit)/10 || digit>=limit) return false;
            v=v*10+digit; if(v>=limit) return false;
        }
        return true;
    };
    for(;;) {
        int a,b; if(!number(a)) {out.clear();return false;} b=a;
        if(p<s.size() && s[p]=='-') {p++;if(!number(b)||b<a){out.clear();return false;}}
        for(int i=a;i<=b;i++) out.push_back(i);
        if(p==s.size()) break;
        if(s[p]==',') {p++;continue;}
        while(p<s.size() && (s[p]=='\n'||s[p]=='\r'||s[p]==' '||s[p]=='\t')) p++;
        if(p!=s.size()){out.clear();return false;} break;
    }
    std::sort(out.begin(),out.end());
    out.erase(std::unique(out.begin(),out.end()),out.end());
    return !out.empty();
}
// Lists must include all topology siblings, even siblings outside the allowed set.
// A partial, unreadable, overlapping or inconsistent topology leaves the old plan.
inline bool plan(const std::vector<int> &allowed, const std::vector<std::string> &lists,
                 int limit, std::vector<int> &representatives) {
    representatives.clear();
    if(allowed.empty() || allowed.size()!=lists.size() || limit<1) return false;
    std::vector<int> sorted=allowed;
    std::sort(sorted.begin(),sorted.end());
    if(std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end()) return false;
    std::vector<std::vector<int>> groups;
    for(std::size_t k=0;k<allowed.size();k++) {
        const int cpu=allowed[k]; std::vector<int> siblings;
        if(cpu<0 || cpu>=limit || !parse_list(lists[k],limit,siblings) ||
           !std::binary_search(siblings.begin(),siblings.end(),cpu)) return false;
        bool found=false;
        for(const auto &g:groups) {
            std::vector<int> intersection;
            std::set_intersection(g.begin(),g.end(),siblings.begin(),siblings.end(),
                                  std::back_inserter(intersection));
            if(!intersection.empty()) {
                if(g!=siblings) return false;
                found=true;break;
            }
        }
        if(!found) groups.push_back(siblings);
    }
    for(const auto &g:groups) {
        auto it=std::find_if(sorted.begin(),sorted.end(),[&](int c){
            return std::binary_search(g.begin(),g.end(),c);
        });
        if(it==sorted.end()) return false;
        representatives.push_back(*it);
    }
    std::sort(representatives.begin(),representatives.end());
    return !representatives.empty();
}
}
