#pragma once

#include <memory>

#include "NeuralNetwork/NodeNetwork.h"
#include "NeuralNetwork/ParameterManager.h"

namespace Training
{
    class BaseTrainer
    {
    public:
        BaseTrainer() = default;
        virtual ~BaseTrainer();

    public:
        // setup
        virtual bool setup();
        virtual bool handleConfigSetup();
        virtual bool setupTrainingData();
        virtual bool setupNetwork();
        virtual NetworkSizeData BaseTrainer::getNetworkSizeData() const;
        virtual bool setupParameters();
        virtual bool handleOptionValidation() const;

    public:
        virtual void run();
        virtual void handleTrainingIteration(int iteration);
        virtual void handleNetworkComputation(int id);
        virtual void handleParamSetEvolution();

    protected:
        virtual std::string getName() const { return "BaseTrainer"; }
        virtual void describeScoreForId(int id) const;
        virtual double computeFinalScore(int id);

    private:
        bool readConfigValues();
        void describeTrainer() const;

    protected:
        // parameters
        ParameterManagerData m_paramData; /// collection of data needed by the parameter manager
        int m_numIterations = 1; /// number of training iterations
        int m_numParamSets = 5; /// number of concurrently tried parameter sets

                                /// number of matches run to compute a score for each parameter set
                                /// actually, we run twice this amount (trying both as first and second player)
        int m_numMatches = 10;

        /// defines the type of activation function 
        /// known types: "relu", "leakyrelu", "sigmoid", "tanh"
        /// anything else is treated as the identity activation function (no modification)
        std::string m_activationFunctionType;

        std::vector<int> m_numHiddenNodes; /// number of nodes within each hidden layer

    protected:
        // internal members
        bool m_initialized = false;

        std::shared_ptr<NeuralNetwork::ParameterManager> m_paramManager;
        std::shared_ptr<NeuralNetwork::NodeNetwork> m_nodeNetwork;
        std::map<int, std::vector<int>> m_idsPerIteration;
    };
}
