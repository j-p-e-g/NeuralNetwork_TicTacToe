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
    int numInputNodes;
    int numOutputNodes;
    std::vector<int> numHiddenNodes;
};

struct ParameterManagerData
{
    int numParams;
    double minValue;
    double maxValue;
};