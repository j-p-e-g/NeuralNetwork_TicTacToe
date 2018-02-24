#pragma once

#include <vector>

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