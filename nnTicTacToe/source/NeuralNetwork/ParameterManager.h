#pragma once

#include <map>
#include <vector>

#include "General/Globals.h"

namespace NeuralNetwork
{
    struct ParamSet
    {
        double score = 0.0;
        bool active = true;
        std::vector<double> params;
    };

    class ParameterManager
    {
    public:
        ParameterManager() = delete;
        ParameterManager(const ParameterManagerData pmData);

    public:
        void describeParameterManager() const;
        bool readDataFromFile();
        bool dumpDataToFile() const;
        void fillWithRandomValues(std::vector<double>& params) const;
        int addNewParamSet(const ParamSet& pset);
        void setScore(int id, double score);
        void setParameterSetActive(int id, bool active);
        void setParameters(int id, const std::vector<double>& pset);
        bool getParamSetForId(int id, ParamSet& pset) const;
        void getActiveParameterSetIds(std::vector<int>& ids) const;
        void getParameterSetIdsSortedByScore(std::vector<int>& bestIds) const;
        void removeParameterSetForId(int id);

        bool evolveParameterSets(int totalNumberOfSets, std::vector<int>& newParameterSetIds);
        void updateEffectiveMutationRates(int newBestSetId);
        double getEffectiveReplacementMutationChance() const { return m_effectiveMutationReplacementChance; }
        double getEffectiveBonusMutationChance() const { return m_effectiveMutationBonusChance; }

        void fillParameterSetProbabilityMap(std::map<double, int> &probabilityMap);
        static int getIdByProbability(const std::map<double, int>& probabilityMap);

        /// create a new parameter set by mutating each parameter in the parent set
        bool createMutatedParameterSet(int id, ParamSet& pset) const;

        /// create a new parameter set by combining the values from two parent sets
        bool createCrossoverParameterSet(int id1, int id2, ParamSet& pset) const;

        double getMutatedValue(double param) const;

    private:
        ParameterManagerData m_paramData;
        std::map<int, ParamSet> m_paramSets;
        int m_nextId;
        bool m_executeMutationStep = true;

        int m_currentBestSetId = -1;
        int m_numIterationsBestIdUnchanged = 0;
        double m_effectiveMutationReplacementChance = 0;
        double m_effectiveMutationBonusChance = 0;
    };
}
