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
    int numParams = 1;
    double minValue = -1;
    double maxValue = 1;
};