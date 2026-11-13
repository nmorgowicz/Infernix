#pragma once
#include "ops/linear/fp8/fp8_geometry.h"

// The K-split A16 tile's schedule (fp8_a16_ksplit_mma.cuh). The GEMV and SIMT schedules, and the
// code-cache and block-order enums the fork keeps beside it, are fp8_schedule.cuh's here.
namespace infernix::ops::detail {

enum class Fp8A16KSplitActivationStage : std::uint8_t {
    ActiveOnly,
    PaddedZero,
};

enum class Fp8A16KSplitCache : std::uint8_t {
    Default,
    Streaming,
};

template <int KWarps, int TileTokens, int MinBlocksPerSm,
          Fp8A16KSplitCache ActivationCache           = Fp8A16KSplitCache::Default,
          Fp8A16KSplitCache WeightCache               = Fp8A16KSplitCache::Streaming,
          Fp8A16KSplitActivationStage ActivationStage = Fp8A16KSplitActivationStage::ActiveOnly>
struct Fp8A16KSplitSchedule {
    static_assert(KWarps == 4 || KWarps == 8 || KWarps == 16);
    static_assert(TileTokens == 8 || TileTokens == 16 || TileTokens == 24 || TileTokens == 32 ||
                  TileTokens == 40 || TileTokens == 48);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kKWarps            = KWarps;
    static constexpr int kTileTokens        = TileTokens;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr auto kActivationCache  = ActivationCache;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kActivationStage  = ActivationStage;
    static constexpr int kThreads           = KWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = KWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / KWarps;
};

} // namespace infernix::ops::detail
