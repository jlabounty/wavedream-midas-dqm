// Runs the real reco headers on the cases generate_nim_pairing_golden.py writes
// and prints what they make of them as JSON (stdout).
//
//   PIPSMSMANimPairing.hh  Prepare() + Pair(diagnostics = true, the wide dt with
//                          the case's widePairsPerFrame budget): one counter
//                          (TOT id 2004, NIM id 2024), plus pass-through hits
//                          (id 9999) that only widen the frame's raw bounds
//   PISMAFineOffset.hh     MakeConfig() + Scan() with one lag channel: the
//                          NIM-copy lag vote (no fallback lag)
//
// Header-only, framework-free; build and run instructions in README.md.
// Input (whitespace separated, one case after the other):
//
//   PAIR <name>
//   <window> <timeSource 0 tot|1 nim> <nimOnlyTot> <totUnit> <echo 0|1>
//   <echoLateTotMin> <echoEdgeTolNs> <totOffset> <nimOffset> <widePairsPerFrame>
//   <n>   then n lines: <role 0 TOT|1 NIM|2 other> <t> <tot> <index>
//
//   LAG <name>
//   <s1 ch> <nim ch> <shift> <lagTolNs> <minPairs> <dominance> <nominal> <useNext 0|1>
//   <n>   then n lines: <ch> <tot> <coarse> <fine>
//
//   END

#include "PIPSMSMANimPairing.hh"
#include "PISMAFineOffset.hh"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kTotVid = 2004;
constexpr int kNimVid = 2024;
constexpr int kOtherVid = 9999;

void num(double x) {
    if (std::isnan(x)) {
        std::printf("null");
    } else {
        std::printf("%.17g", x);
    }
}

void lag(std::int32_t v) {
    if (v == PISMAWord::kNoLag) {
        std::printf("null");
    } else {
        std::printf("%d", v);
    }
}

bool pairCase(std::istream& in, const std::string& name) {
    namespace P = PIPSMSMANimPairing;
    double window, nimOnlyTot, totUnit, edgeTol, totOff, nimOff;
    int timeSource, echo, lateMin;
    std::size_t wideBudget;
    in >> window >> timeSource >> nimOnlyTot >> totUnit >> echo >> lateMin >> edgeTol >> totOff >> nimOff
        >> wideBudget;
    std::size_t n;
    in >> n;
    std::vector<P::RawHit> hits(n);
    std::vector<int> roles(n);
    for (std::size_t i = 0; i < n; ++i) {
        int role;
        in >> role >> hits[i].t >> hits[i].tot >> hits[i].index;
        roles[i] = role;
        hits[i].vid = role == 0 ? kTotVid : role == 1 ? kNimVid : kOtherVid;
    }
    if (!in) return false;

    P::Config cfg;
    cfg.pairs[kTotVid] = kNimVid;
    cfg.offsetNs[kTotVid] = totOff;
    cfg.offsetNs[kNimVid] = nimOff;
    cfg.cabledNim.insert(kNimVid);
    cfg.pairWindowNs = window;
    cfg.timeSource = timeSource ? P::TimeSource::kNim : P::TimeSource::kTot;
    cfg.nimOnlyTot = static_cast<float>(nimOnlyTot);
    cfg.totUnitNs = totUnit;
    if (echo) cfg.echoVids.insert(kTotVid);
    cfg.echoLateTotMin = lateMin;
    cfg.echoEdgeTolNs = edgeTol;
    cfg.widePairsPerFrame = wideBudget;
    P::Setup setup;
    std::string why;
    if (!P::Prepare(cfg, setup, &why)) {
        std::cerr << name << ": " << why << "\n";
        return false;
    }
    P::FrameResult out;
    P::Pair(setup, hits, out, true);

    std::printf("{\"kind\": \"pair\", \"name\": \"%s\",\n \"config\": {\"window\": ", name.c_str());
    num(window);
    std::printf(", \"time_source\": \"%s\", \"nim_only_tot\": ", timeSource ? "nim" : "tot");
    num(nimOnlyTot);
    std::printf(", \"tot_unit\": ");
    num(totUnit);
    std::printf(", \"echo\": %s, \"echo_late_tot\": %d, \"echo_edge_tol\": ", echo ? "true" : "false", lateMin);
    num(edgeTol);
    std::printf(", \"tot_offset\": ");
    num(totOff);
    std::printf(", \"nim_offset\": ");
    num(nimOff);
    std::printf(", \"wide_budget\": %zu},\n \"hits\": [", wideBudget);
    for (std::size_t i = 0; i < n; ++i) {
        std::printf("%s[%d, ", i ? ", " : "", roles[i]);
        num(hits[i].t);
        std::printf(", %d, %d]", hits[i].tot, hits[i].index);
    }
    std::printf("],\n \"merged\": [");
    bool first = true;
    for (const P::MergedHit& m : out.hits) {
        if (m.vid != kTotVid) continue;   // pass-through hits
        std::printf("%s\n  {\"time\": ", first ? "" : ",");
        first = false;
        num(m.time);
        std::printf(", \"t_tot\": ");
        num(m.tTot);
        std::printf(", \"t_nim\": ");
        num(m.tNim);
        std::printf(", \"tot\": %d, \"nim_width\": %d, \"flags\": %u, \"raw_tot\": %d, \"raw_nim\": %d, \"edep\": ",
                    m.tot, m.nimWidth, static_cast<unsigned>(m.flags), m.rawTotIndex, m.rawNimIndex);
        num(m.edep);
        std::printf("}");
    }
    const P::CounterCounts& c = out.counts.at(0);
    std::printf("],\n \"counts\": {\"tot_words\": %llu, \"nim_words\": %llu, \"paired\": %llu, \"tot_only\": %llu, "
                "\"nim_only\": %llu, \"echo\": %llu, \"shadow\": %llu, \"multi\": %llu, \"near_edge\": %llu},\n",
                static_cast<unsigned long long>(c.totWords), static_cast<unsigned long long>(c.nimWords),
                static_cast<unsigned long long>(c.paired), static_cast<unsigned long long>(c.totOnly),
                static_cast<unsigned long long>(c.nimOnly), static_cast<unsigned long long>(c.echo),
                static_cast<unsigned long long>(c.shadow), static_cast<unsigned long long>(c.multi),
                static_cast<unsigned long long>(c.nearEdge));
    std::printf(" \"nim_diag\": [");
    for (std::size_t i = 0; i < out.nim.size(); ++i) {
        std::printf("%s[", i ? ", " : "");
        num(out.nim[i].dtAligned);
        std::printf(", %d, %d]", out.nim[i].nearestTot, out.nim[i].nimWidth);
    }
    std::printf("],\n \"tot_diag\": [");
    for (std::size_t i = 0; i < out.tot.size(); ++i) std::printf("%s%d", i ? ", " : "", out.tot[i].candidates);
    std::printf("],\n \"pair_diag\": [");
    for (std::size_t i = 0; i < out.pairs.size(); ++i) {
        std::printf("%s[", i ? ", " : "");
        num(out.pairs[i].dt);
        std::printf(", %d]", out.pairs[i].tot);
    }
    std::printf("],\n \"wide\": [");
    for (std::size_t i = 0; i < out.wide.size(); ++i) {
        std::printf("%s", i ? ", " : "");
        num(out.wide[i].dt);
    }
    std::printf("]}");
    return true;
}

bool lagCase(std::istream& in, const std::string& name) {
    namespace F = PISMAFineOffset;
    int s1, nim, minPairs, nominal, useNext;
    unsigned shift;
    double tolNs, dominance;
    in >> s1 >> nim >> shift >> tolNs >> minPairs >> dominance >> nominal >> useNext;
    std::size_t n;
    in >> n;
    std::vector<std::uint64_t> words(n);
    std::vector<std::array<std::uint64_t, 4>> fields(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t ch, tot, coarse, fine;
        in >> ch >> tot >> coarse >> fine;
        fields[i] = {ch, tot, coarse, fine};
        words[i] = (std::uint64_t{1} << 63) | ((ch & 0xF) << 56) | ((tot & 0xFF) << 48)
                   | ((coarse & 0xFFFFFFF) << 20) | (fine & 0xFFFFF);
    }
    if (!in) return false;
    const std::array<int, 1> lagCh{nim};
    const std::array<std::int32_t, 1> lagNominal{nominal};
    const auto cfg = F::MakeConfig(shift, s1, {}, {}, {}, lagCh, lagNominal, 50.0, tolNs,
                                   static_cast<std::uint32_t>(minPairs), dominance, 0.05, 49152, useNext != 0);
    if (!cfg) {
        std::cerr << name << ": MakeConfig refused the settings\n";
        return false;
    }
    std::array<int, F::kChannels> fallback;
    fallback.fill(-1);
    fallback[static_cast<std::size_t>(nim)] = PISMAWord::kNoLag;
    F::BankResult out;
    F::LagScratch scratch;
    F::Scan(words.data(), words.data() + words.size(), *cfg, fallback, out, &scratch);
    const F::ChannelResult& r = out.ch[static_cast<std::size_t>(nim)];
    std::vector<std::uint32_t> d = scratch.d[0];   // sorted in place by FindLagModes
    std::vector<std::uint32_t> dd = d;
    const PISMAWord::LagModes modes = PISMAWord::FindLagModes(dd, static_cast<std::uint32_t>(2.0 * tolNs));

    std::printf("{\"kind\": \"lag\", \"name\": \"%s\",\n \"config\": {\"s1\": %d, \"nim\": %d, \"shift\": %u, "
                "\"lag_tol\": ", name.c_str(), s1, nim, shift);
    num(tolNs);
    std::printf(", \"min_pairs\": %d, \"dominance\": ", minPairs);
    num(dominance);
    std::printf(", \"nominal\": %d, \"next_reference\": %s},\n \"words\": [", nominal, useNext ? "true" : "false");
    for (std::size_t i = 0; i < n; ++i) {
        std::printf("%s[%llu, %llu, %llu, %llu]", i ? ", " : "", static_cast<unsigned long long>(fields[i][0]),
                    static_cast<unsigned long long>(fields[i][1]), static_cast<unsigned long long>(fields[i][2]),
                    static_cast<unsigned long long>(fields[i][3]));
    }
    std::printf("],\n \"n_pairs\": %u, \"source\": %d, \"lag\": ", r.nPairs, static_cast<int>(r.source));
    lag(r.lagNs);
    std::printf(", \"runner_up_lag\": ");
    lag(r.lagRunnerUpNs);
    std::printf(", \"best_lag\": ");
    lag(modes.best.n > 0 ? modes.best.lagNs : PISMAWord::kNoLag);
    std::printf(", \"n_best\": %u, \"n_runner_up\": %u, \"shift_ns\": %d,\n \"d_sorted\": [", r.nBest, r.nRunnerUp,
                r.lagShiftNs);
    for (std::size_t i = 0; i < d.size(); ++i) std::printf("%s%u", i ? ", " : "", d[i]);
    std::printf("]}");
    return true;
}

}  // namespace

int main() {
    std::string tag, name;
    bool first = true;
    std::printf("[");
    while (std::cin >> tag && tag != "END") {
        std::cin >> name;
        std::printf("%s\n", first ? "" : ",");
        first = false;
        const bool ok = tag == "PAIR" ? pairCase(std::cin, name) : tag == "LAG" ? lagCase(std::cin, name) : false;
        if (!ok) {
            std::cerr << "bad case " << tag << " " << name << "\n";
            return 1;
        }
    }
    std::printf("\n]\n");
    return 0;
}
