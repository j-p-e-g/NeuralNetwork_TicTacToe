#pragma once

#include <memory>
#include <vector>

#include "Game/GameLogic.h"
#include "Game/Player.h"
#include "NeuralNetwork/NodeNetwork.h"
#include "NeuralNetwork/ParameterManager.h"

namespace Training
{
    class TrainingMethodHandler
    {
    public:
        TrainingMethodHandler(std::shared_ptr<NeuralNetwork::NodeNetwork>& network, std::shared_ptr<NeuralNetwork::ParameterManager>& paramManager, std::shared_ptr<Game::GameLogic>& gameLogic);
        virtual ~TrainingMethodHandler() = default;

    public:
        virtual std::string getName() const { return "TrainingMethodHandler"; }
        virtual void describeTrainingMethod() const;
        virtual void iterationStart(int paramSetId);
        virtual double handleTrainingIteration(std::shared_ptr<Game::BasePlayer>& player) = 0;
        virtual void iterationEnd(bool lastIteration) = 0;
        virtual void postIteration(bool lastIteration = false) = 0;

    protected:
        std::shared_ptr<NeuralNetwork::NodeNetwork> m_nodeNetwork;
        std::shared_ptr<NeuralNetwork::ParameterManager> m_paramManager;
        std::shared_ptr<Game::GameLogic> m_gameLogic;
        int m_currentParamSetId;
    };

    class ParameterEvolutionHandler
        : public TrainingMethodHandler
    {
    public:
        ParameterEvolutionHandler(std::shared_ptr<NeuralNetwork::NodeNetwork>& network, std::shared_ptr<NeuralNetwork::ParameterManager>& paramManager, std::shared_ptr<Game::GameLogic>& gameLogic);
        virtual ~ParameterEvolutionHandler() = default;

    public:
        std::string getName() const override { return "ParameterEvolutionHandler"; }
        double handleTrainingIteration(std::shared_ptr<Game::BasePlayer>& player) override;
        void iterationEnd(bool lastIteration) override;
        void postIteration(bool lastIteration = false) override;

    private:
    };


    class BackpropagationHandler
        : public TrainingMethodHandler
    {
    public:
        BackpropagationHandler(std::shared_ptr<NeuralNetwork::NodeNetwork>& network, std::shared_ptr<NeuralNetwork::ParameterManager>& paramManager, std::shared_ptr<Game::GameLogic>& gameLogic);
        virtual ~BackpropagationHandler() = default;

    public:
        std::string getName() const override { return "BackpropagationHandler"; }
        void describeTrainingMethod() const override;
        void iterationStart(int paramSetId) override;
        double handleTrainingIteration(std::shared_ptr<Game::BasePlayer>& player) override;
        void iterationEnd(bool lastIteration) override;
        void postIteration(bool lastIteration = false) override;

    private:
        std::vector<double> m_parameterAdjustmentValues;
        std::vector<double> m_errors;

        double m_learningRate = 10;
        double m_learningRateFactor = 0.5;
        double m_minLearningRate = 0.0001;
        double m_maxLearningRate = 10;
        /// if true, the averaged adjustment values get divided by the largest entry
        /// so their proportion stays the same, but large values don't throw the algorithm out of whack
        bool m_normalizeAdjustmentValues = true;

        int m_countTrainingSets = 0;
        double m_prevError = INFINITY;
    };
}
