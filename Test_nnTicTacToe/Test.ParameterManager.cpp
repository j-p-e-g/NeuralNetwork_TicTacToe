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
            data.minValue = -2;
            data.maxValue = 5;

            ParameterManager pm(data);

            // assigned the correct number of values
            std::vector<double> params;
            pm.fillWithRandomValues(params);
            Assert::AreEqual(data.numParams, static_cast<int>(params.size()));

            // values are within boundaries
            std::set<double> uniqueParams;
            for (const auto& val : params)
            {
                Assert::AreEqual(true, val >= data.minValue);
                Assert::AreEqual(true, val <= data.maxValue);

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
            data.minValue = -1.0;
            data.maxValue = 1.0;

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

        TEST_METHOD(ParameterManager_getParameterSetIdsSortedByScore)
        {
            ParameterManagerData data;
            data.numParams = 3;
            data.minValue = 0;
            data.maxValue = 10;

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
    };
}
