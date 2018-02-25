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
    };
}
