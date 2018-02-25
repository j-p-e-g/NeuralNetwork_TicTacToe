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
        ParameterManager(const ParameterManagerData pmData);

    public:
        bool readDataFromFile();
        bool dumpDataToFile() const;
        void fillWithRandomValues(std::vector<double>& params) const;
        void addNewParamSet(const ParamSet& pset);
        void setScore(int id, double score);
        void setParameters(int id, const std::vector<double>& pset);
        bool getParamSetForId(int id, ParamSet& pset) const;
        void getParameterSetIdsSortedByScore(std::vector<int>& bestIds) const;

    private:
        ParameterManagerData m_paramData;
        std::map<int, ParamSet> m_paramSets;
        int m_nextId;
    };
}
