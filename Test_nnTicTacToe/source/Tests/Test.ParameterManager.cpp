#include "stdafx.h"
#include "CppUnitTest.h"
#include "NeuralNetwork/ParameterManager.h"

#include <set>

namespace ParameterManagerTest
{
    using namespace Microsoft::VisualStudio::CppUnitTestFramework;
    using namespace NeuralNetwork;

    TEST_CLASS(ParameterManager_Test)
    {
    public:
        // -----------------------------------------
        // ParameterManager
        // -----------------------------------------
        TEST_METHOD(ParameterManager_fillWithRandomValues)
        {
            ParameterManagerData data;
            data.numParams = 10;
            data.minRandomParamValue = -2;
            data.maxRandomParamValue = 5;

            ParameterManager pm(data);

            // assigned the correct number of values
            std::vector<double> params;
            pm.fillWithRandomValues(params);
            Assert::AreEqual(data.numParams, static_cast<int>(params.size()));

            // values are within boundaries
            std::set<double> uniqueParams;
            for (const auto& val : params)
            {
                Assert::AreEqual(true, val >= data.minRandomParamValue);
                Assert::AreEqual(true, val <= data.maxRandomParamValue);

                uniqueParams.emplace(val);
            }

            // not all the same value
            // in theory this test could fail, but it's extremely unlikely
            Assert::AreEqual(true, uniqueParams.size() > 1);
        }

        TEST_METHOD(ParameterManager_getParamSetForId)
        {
            ParameterManagerData data;
            data.numParams = 2;
            data.minRandomParamValue = -1.0;
            data.maxRandomParamValue = 1.0;

            ParameterManager pm(data);

            // first id is 0
            ParamSet pset;
            pset.score = 57.45;
            pset.params = std::vector<double>({ 0.8, -1.3 });
            pm.addNewParamSet(pset);

            ParamSet pset2;
            pm.getParamSetForId(0, pset2);

            // all values are equal to what was initially entered
            Assert::AreEqual(57.45, pset2.score, 0.0001);
            Assert::AreEqual(2, static_cast<int>(pset2.params.size()));
            Assert::AreEqual(0.8, pset2.params[0], 0.0001);
            Assert::AreEqual(-1.3, pset2.params[1], 0.0001);
        }

        TEST_METHOD(ParameterManager_getParameterSetIds)
        {
            ParameterManagerData data;
            data.numParams = 3;
            data.minRandomParamValue = 1;
            data.maxRandomParamValue = 2;

            ParameterManager pm(data);

            ParamSet pset1;
            pm.addNewParamSet(pset1);

            ParamSet pset2;
            pm.addNewParamSet(pset2);

            ParamSet pset3;
            pm.addNewParamSet(pset3);

            pm.removeParameterSetForId(0);

            std::vector<int> paramIds;
            pm.getActiveParameterSetIds(paramIds);

            Assert::AreEqual(2, static_cast<int>(paramIds.size()));
            Assert::AreEqual(true, std::find(paramIds.begin(), paramIds.end(), 1) != paramIds.end());
            Assert::AreEqual(true, std::find(paramIds.begin(), paramIds.end(), 2) != paramIds.end());
        }

        TEST_METHOD(ParameterManager_getParameterSetIdsSortedByScore)
        {
            ParameterManagerData data;
            data.numParams = 3;
            data.minRandomParamValue = 0;
            data.maxRandomParamValue = 10;

            ParameterManager pm(data);

            // score order: pset2 (id 1) > pset3 (id 2) > pset1 (id 0)
            ParamSet pset1;
            pset1.score = 31.2;
            pset1.params = std::vector<double>({ 7.2, 1.345, 9.9 });
            pm.addNewParamSet(pset1);

            ParamSet pset2;
            pset2.score = 47.003;
            pset2.params = std::vector<double>({ 2.0, 3.0, 5.0 });
            pm.addNewParamSet(pset2);

            ParamSet pset3;
            pset3.score = 46.9999;
            pset3.params = std::vector<double>({ 8.9, 1.0, 4.32 });
            pm.addNewParamSet(pset3);

            std::vector<int> sortedIds;
            pm.getParameterSetIdsSortedByScore(sortedIds);

            // assert the correct order
            Assert::AreEqual(3, static_cast<int>(sortedIds.size()));
            Assert::AreEqual(1, sortedIds[0]);
            Assert::AreEqual(2, sortedIds[1]);
            Assert::AreEqual(0, sortedIds[2]);

            // compare the values for the best-score id with its initial values
            ParamSet bestSet;
            pm.getParamSetForId(sortedIds[0], bestSet);

            Assert::AreEqual(47.003, bestSet.score, 0.0001);
            Assert::AreEqual(3, static_cast<int>(bestSet.params.size()));
            Assert::AreEqual(2.0, bestSet.params[0], 0.0001);
            Assert::AreEqual(3.0, bestSet.params[1], 0.0001);
            Assert::AreEqual(5.0, bestSet.params[2], 0.0001);
        }

        TEST_METHOD(ParameterManager_fillParameterSetProbabilityMap)
        {
            ParameterManagerData data;
            data.numParams = 4;
            data.minRandomParamValue = -5;
            data.maxRandomParamValue = 5;

            ParameterManager pm(data);

            ParamSet ps1;
            ps1.params = { 5, 2.1, 3, 4 };
            ps1.score = 8;

            ParamSet ps2;
            ps2.params = { 0.9, 4.8, -2.2, 0 };
            ps2.score = -2;

            ParamSet ps3;
            ps3.params = { -4, 3, 1.21, -0.5 };
            ps3.score = 4;

            pm.addNewParamSet(ps1); // id 0
            pm.addNewParamSet(ps2); // id 1
            pm.addNewParamSet(ps3); // id 2

            std::map<double, int> probabilityMap;
            pm.fillParameterSetProbabilityMap(probabilityMap);
            Assert::AreEqual(2, static_cast<int>(probabilityMap.size()));
            
            // 10/16 = 0.625
            const auto& found1 = probabilityMap.find(0.625);
            Assert::AreEqual(true, found1 != probabilityMap.end());
            Assert::AreEqual(0, found1->second);

            const auto& found2 = probabilityMap.find(1.0);
            Assert::AreEqual(true, found2 != probabilityMap.end());
            Assert::AreEqual(2, found2->second);
        }

        TEST_METHOD(ParameterManager_getIdByProbability)
        {
            std::map<double, int> probabilityMap;
            probabilityMap.emplace(0.2, 5);
            probabilityMap.emplace(1.0, 19);

            std::map<int, int> candidates;
            for (int k = 0; k < 20; k++)
            {
                const int result = ParameterManager::getIdByProbability(probabilityMap);
                Assert::AreNotEqual(-1, result);

                auto& found = candidates.find(result);
                if (found == candidates.end())
                {
                    candidates.emplace(result, 1);
                }
                else
                {
                    found->second++;
                }
            }

            const auto& found1 = candidates.find(5);
            const auto& found2 = candidates.find(19);

            // may fail but that's not very likely
            Assert::AreEqual(true, found1->second < found2->second);
        }

        TEST_METHOD(ParameterManager_updateEffectiveMutationRates_default)
        {
            ParameterManagerData data;
            data.mutationReplacementChance = 0.0025;
            data.mutationBonusChance = 0.03;
            data.mutationBonusScale = 0.7;
            data.maxMutationBonusChance = 0.8;
            data.maxMutationReplacementChance = 0.45;
            data.mutationRateIterationMultiplier = 0.1;

            // by default, the effective rates should be identical to the base rates
            ParameterManager pm(data);
            Assert::AreEqual(data.mutationReplacementChance, pm.getEffectiveReplacementMutationChance(), 0.0001);
            Assert::AreEqual(data.mutationBonusChance, pm.getEffectiveBonusMutationChance(), 0.0001);
        }

        TEST_METHOD(ParameterManager_updateEffectiveMutationRates_sameId)
        {
            ParameterManagerData data;
            data.mutationReplacementChance = 0.02;
            data.mutationBonusChance = 0.1;
            data.mutationBonusScale = 1;
            data.maxMutationBonusChance = 0.5;
            data.maxMutationReplacementChance = 0.2;
            data.mutationRateIterationMultiplier = 0.5;

            ParameterManager pm(data);

            // fake iterations with the same best set id 3 times in a row
            for (int k = 0; k < 4; k++)
            {
                pm.updateEffectiveMutationRates(5);
            }

            Assert::AreEqual(0.05, pm.getEffectiveReplacementMutationChance(), 0.0001);
            Assert::AreEqual(0.25, pm.getEffectiveBonusMutationChance(), 0.0001);

            // fake some more iterations with the same best set
            for (int k = 0; k < 50; k++)
            {
                pm.updateEffectiveMutationRates(5);
            }

            // capped by maximum
            Assert::AreEqual(data.maxMutationReplacementChance, pm.getEffectiveReplacementMutationChance(), 0.0001);
            Assert::AreEqual(data.maxMutationBonusChance, pm.getEffectiveBonusMutationChance(), 0.0001);
        }

        TEST_METHOD(ParameterManager_updateEffectiveMutationRates_idChanged)
        {
            ParameterManagerData data;
            data.mutationReplacementChance = 0.01;
            data.mutationBonusChance = 0.05;
            data.mutationBonusScale = 0.5;
            data.maxMutationReplacementChance = 0.1;
            data.maxMutationBonusChance = 0.25;
            data.mutationRateIterationMultiplier = 0.5;

            ParameterManager pm(data);

            // fake iterations with the same set id a couple of times
            for (int k = 0; k < 5; k++)
            {
                pm.updateEffectiveMutationRates(19);
            }

            // change to a different "best id"
            pm.updateEffectiveMutationRates(2);

            // should be back at the initial values
            Assert::AreEqual(data.mutationReplacementChance, pm.getEffectiveReplacementMutationChance(), 0.0001);
            Assert::AreEqual(data.mutationBonusChance, pm.getEffectiveBonusMutationChance(), 0.0001);
        }

        TEST_METHOD(ParameterManager_getMutatedValue_noMutation_chances)
        {
            ParameterManagerData data;
            data.minRandomParamValue = -25;
            data.maxRandomParamValue = 25;
            data.mutationBonusChance = 0;
            data.mutationReplacementChance = 0;
            data.mutationBonusScale = 10;

            ParameterManager pm(data);

            // if the mutation chance is zero, the value should stay unchanged
            for (int k = 0; k < 20; k++)
            {
                Assert::AreEqual(k, pm.getMutatedValue(k), 0.0001);
            }
        }

        TEST_METHOD(ParameterManager_getMutatedValue_noMutation_BonusScale)
        {
            ParameterManagerData data;
            data.minRandomParamValue = -3;
            data.maxRandomParamValue = 3;
            data.mutationReplacementChance = 0;
            data.mutationBonusChance = 1;
            data.mutationBonusScale = 0;

            ParameterManager pm(data);

            // if the bonus scale is zero, the value should stay unchanged
            for (int k = -10; k < 10; k++)
            {
                Assert::AreEqual(k, pm.getMutatedValue(k), 0.0001);
            }
        }

        TEST_METHOD(ParameterManager_getMutatedValue_alwaysReplace)
        {
            ParameterManagerData data;
            data.minRandomParamValue = -5;
            data.maxRandomParamValue = 5;
            data.mutationReplacementChance = 1;
            data.mutationBonusChance = 0;
            data.mutationBonusScale = 10;

            ParameterManager pm(data);

            // if the replacement chance is 1, the value will always be randomly picked within [min, max]
            std::vector<double> randomValues;
            for (int k = 30; k < 50; k++)
            {
                randomValues.push_back(pm.getMutatedValue(k));
            }

            std::set<double> uniqueValues;
            for (const auto& val : randomValues)
            {
                Assert::AreEqual(true, val >= -5);
                Assert::AreEqual(true, val <= 5);

                uniqueValues.emplace(val);
            }

            // not all the same value
            // may fail but that's extremely unlikely
            Assert::AreEqual(true, uniqueValues.size() > 1);
        }

        TEST_METHOD(ParameterManager_getMutatedValue_replacementChance)
        {
            ParameterManagerData data;
            data.minRandomParamValue = -20;
            data.maxRandomParamValue = -10;
            data.mutationReplacementChance = 0.5;
            data.mutationBonusChance = 0;
            data.mutationBonusScale = 0;

            ParameterManager pm(data);

            int countReplaced = 0;
            for (int k = 0; k < 100; k++)
            {
                const double mutatedValue = pm.getMutatedValue(k);
                if (mutatedValue != k)
                {
                    Assert::AreEqual(true, mutatedValue >= -20);
                    Assert::AreEqual(true, mutatedValue <= -10);

                    countReplaced++;
                }
            }

            // some but not all values got replaced
            // this may fail but is highly unlikely
            Assert::AreNotEqual(0, countReplaced);
            Assert::AreNotEqual(100, countReplaced);
        }

        TEST_METHOD(ParameterManager_getMutatedValue_noBonusChance)
        {
            ParameterManagerData data;
            data.minRandomParamValue = -8;
            data.maxRandomParamValue = 8;
            data.mutationReplacementChance = 0.1;
            data.mutationBonusChance = 0;
            data.mutationBonusScale = 10;

            ParameterManager pm(data);

            // if the bonus chance is zero, the value may be replaced (replacement chance) but it won't get a bonus
            for (int k = 100; k < 120; k++)
            {
                const double mutatedValue = pm.getMutatedValue(k);
                Assert::AreEqual(true, mutatedValue == k || mutatedValue >= -8 && mutatedValue <= 8);
            }
        }

        TEST_METHOD(ParameterManager_getMutatedValue_alwaysBonus)
        {
            ParameterManagerData data;
            data.minRandomParamValue = -10;
            data.maxRandomParamValue = 10;
            data.mutationReplacementChance = 0;
            data.mutationBonusChance = 1;
            data.mutationBonusScale = 0.25;

            ParameterManager pm(data);

            // if the replacement chance is zero and the bonus chance is 1, 
            // the value will always get a random bonus from within [min/4, max/4]
            // and be capped between [min, max]
            std::vector<double> randomValues;
            for (int k = -10; k < 10; k++)
            {
                const double mutatedValue = pm.getMutatedValue(k);

                // exclude -10 and 10 in this check as with those it's too likely for this test to fail 
                // (due to capping the mutated value)
                if (k > -10 && k < 10)
                {
                    Assert::AreNotEqual(k, mutatedValue, 0.0001);
                }

                Assert::AreEqual(true, mutatedValue >= k - 2.5);
                Assert::AreEqual(true, mutatedValue <= k + 2.5);
                Assert::AreEqual(true, mutatedValue >= -10);
                Assert::AreEqual(true, mutatedValue <= 10);
            }
        }

        TEST_METHOD(ParameterManager_getMutatedValue_bonusChance)
        {
            ParameterManagerData data;
            data.numParams = 100;
            data.minRandomParamValue = -20;
            data.maxRandomParamValue = 20;
            data.mutationReplacementChance = 0;
            data.mutationBonusChance = 0.3;
            data.mutationBonusScale = 1;

            ParameterManager pm(data);

            std::vector<double> values;
            pm.fillWithRandomValues(values);

            int countChanged = 0;
            for (const auto& val : values)
            {
                const double mutatedValue = pm.getMutatedValue(val);
                if (mutatedValue != val)
                {
                    countChanged++;

                    Assert::AreEqual(true, mutatedValue >= -20);
                    Assert::AreEqual(true, mutatedValue <= 20);
                    Assert::AreEqual(true, mutatedValue >= val - 20);
                    Assert::AreEqual(true, mutatedValue <= val + 20);
                }
            }

            // some but not all values got changed
            // this may fail but is rather unlikely
            Assert::AreNotEqual(0, countChanged);
            Assert::AreNotEqual(data.numParams, countChanged);
        }

        TEST_METHOD(ParameterManager_createCrossOverParameterSet)
        {
            ParamSet ps1;
            ps1.params = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };

            ParamSet ps2;
            ps2.params = { -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20 };

            ParameterManagerData data;
            data.numParams = 10;
            data.minRandomParamValue = -0.5;
            data.maxRandomParamValue = 0.5;

            ParameterManager pm(data);
            pm.addNewParamSet(ps1);
            pm.addNewParamSet(ps2);

            ParamSet newParamSet;
            Assert::AreEqual(true, pm.createCrossoverParameterSet(1, 0, newParamSet));

            int countPositive = 0;
            int countNegative = 0;
            int countRandom = 0;
            for (int k = 0; k < 10; k++)
            {
                const double absValue = std::abs(newParamSet.params[k]);
                // either taken from param set 1 or 2, or a new value within [-0.5, 0.5]
                Assert::AreEqual(true, absValue >= -0.5 && absValue <= 0.5 || absValue == k+1);

                if (absValue == k+1)
                {
                    if (newParamSet.params[k] > 0)
                    {
                        countPositive++;
                    }
                    else
                    {
                        countNegative++;
                    }
                }
                else
                {
                    countRandom++;
                }
            }

            // NOTE: any of these might fail, although with low probability
            Assert::AreEqual(true, countRandom < countNegative);
            Assert::AreEqual(true, countRandom < countPositive);
            Assert::AreEqual(true, countNegative > 0);
            Assert::AreEqual(true, countPositive > 0);
        }

        TEST_METHOD(ParameterManager_evolveParameterSets)
        {
            ParameterManagerData data;
            data.numParams = 2;
            data.minRandomParamValue = 0;
            data.maxRandomParamValue = 5;

            ParameterManager pm(data);

            ParamSet ps1; // id 0
            ps1.params = { -2, -2 };
            ps1.score = 5;

            ParamSet ps2; // id 1
            ps2.params = { -4, -4 };
            ps2.score = 10;

            ParamSet ps3; // id 2
            ps3.params = { -6, -6 };
            ps3.score = 3;

            ParamSet ps4; // id 3
            ps4.params = { -8, -8 };
            ps4.score = 2;

            pm.addNewParamSet(ps1);
            pm.addNewParamSet(ps2);
            pm.addNewParamSet(ps3);
            pm.addNewParamSet(ps4);

            std::vector<int> newParameterSetIds;
            Assert::AreEqual(true, pm.evolveParameterSets(4, newParameterSetIds));

            // definitely contains the most highly scoring set (id 1)
            Assert::AreEqual(true, std::find(newParameterSetIds.begin(), newParameterSetIds.end(), 1) != newParameterSetIds.end());

            // the others should be new ids
            for (auto id : newParameterSetIds)
            {
                ParamSet pset;
                pm.getParamSetForId(id, pset);

                if (id == 1)
                {
                    continue;
                }

                // new entry (uninitialized)
                Assert::AreEqual(true, id > 3);
                Assert::AreEqual(0.0, pset.score, 0.0001);

                // parameters are copied either from an existing value (-2, -4, -6) or randomly picked from [0, 5]
                Assert::AreEqual(2, static_cast<int>(pset.params.size()));
                Assert::AreEqual(true, pset.params[0] >= 0 && pset.params[0] <= 5 || pset.params[0] >= -6 && pset.params[0] <= -2);
                Assert::AreEqual(true, pset.params[1] >= 0 && pset.params[1] <= 5 || pset.params[1] >= -6 && pset.params[1] <= -2);
            }
        }
    };
}
