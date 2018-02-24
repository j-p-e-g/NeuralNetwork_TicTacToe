#pragma once

#include <map>
#include <vector>

#include "General/Globals.h"

namespace NeuralNetwork
{
    struct ParamSet
    {
        double score = 0.0;
        std::vector<double> params;
    };

    class ParameterManager
    {
    public:
        ParameterManager() = delete;
        ParameterManager(const ParameterManagerData& pmData);

    public:
        bool readDataFromFile();
        bool dumpDataToFile() const;
        void fillWithRandomValues(ParamSet& pset) const;
        void addNewParamSet(const ParamSet& pset);

    private:
        const ParameterManagerData& paramData;
        std::map<int, ParamSet> m_paramSets;
        int m_highestId;
    };
}
