#pragma once

#include <vector>

enum GameState
{
    GS_ONGOING,
    GS_INVALID,
    GS_GAMEOVER_WON,
    GS_GAMEOVER_LOST,
    GS_GAMEOVER_TIMEOUT
};

enum CellState
{
    CS_EMPTY = 0,
    CS_PLAYER1 = 1,
    CS_PLAYER2 = -1
};

struct NetworkSizeData
{
    int numInputNodes = 1;
    int numOutputNodes = 1;
    std::vector<int> numHiddenNodes;
};

struct ParameterManagerData
{
    int numParamSets = 10; /// number of concurrently tried parameter sets
    int numParams = 1; /// total number of parameters per set
    double minRandomParamValue = -1; /// lower limit for randomly picked values
    double maxRandomParamValue = 1; /// upper limit for randomly picked values

    /// chance of a single parameter being replaced with a random value
    /// in the parameter set evolution step
    double mutationReplacementChance = 0.01;

    /// chance of a single parameter receiving a bonus
    /// in the parameter set evolution step
    double mutationBonusChance = 0.01;

    /// how much the [min, max] range is scaled when a random bonus is added to a parameter
    /// in the parameter set evolution step
    double mutationBonusScale = 1;

    /// defines how much the mutation rates are increased each iteration that the best id doesn't change
    double mutationRateIterationMultiplier = 1;

    /// the maximum the mutation bonus chance can reach after lots of iterations without improvement
    double maxMutationBonusChance = 1;

    /// the maximum the mutation replacement chance can reach after lots of iterations without improvement
    double maxMutationReplacementChance = 1;

    int numBestSetsKeptDuringEvolution = 1; /// number of the top parameter sets (sorted by score) copied over from the previous iteration
    int numBestSetsMutatedDuringEvolution = 1; /// number of sets newly created by mutating the best sets from the previous iteration
    int numAddedRandomSetsDuringEvolution = 1; /// number of random sets newly added at each evolution step
};