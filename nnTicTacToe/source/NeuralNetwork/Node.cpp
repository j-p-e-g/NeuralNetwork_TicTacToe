#include "stdafx.h"

#include "assert.h"

#include "Node.h"

namespace NeuralNetwork
{
    // --------------------------
    // Edge
    // --------------------------
    Edge::Edge(const std::shared_ptr<Node>& pred)
        : m_predecessor(pred)
        , m_edgeWeight(1.f)
    {
    }

    Edge::~Edge()
    {
        m_predecessor = nullptr;
    }

    bool Edge::assignParameters(std::queue<double>& params)
    {
        if (static_cast<int>(params.size()) < getNumParameters())
        {
            return false;
        }

        setEdgeWeight(params.front());
        params.pop();
        return true;
    }

    void Edge::getParameters(std::vector<double>& values) const
    {
        values.push_back(m_edgeWeight);
    }

    int Edge::getNumParameters() const
    {
        return 1;
    }

    double Edge::getValue() const
    {
        return m_predecessor->getValue() * m_edgeWeight;
    }

    double Edge::handleBackpropagation(double multiplier)
    {
        const double inputValueAdjustment = m_edgeWeight * multiplier;
        m_edgeWeight = m_predecessor->getValue() * multiplier;

        return inputValueAdjustment;
    }

    // --------------------------
    // Node
    // --------------------------
    Node::Node()
        : m_value(0.f)
    {
    }

    bool Node::assignParameters(std::queue<double>& params)
    {
        // nothing to do
        return true;
    }

    void Node::getParameters(std::vector<double>& params) const
    {
        // nothing to do
    }

    void Node::updateValue(std::function<double(double, bool)> activationFunction)
    {
        // nothing to do
    }

    void Node::handleBackpropagation(double targetValue, std::function<double(double, bool)> activationFunction, std::vector<double>& inputValueAdjustments)
    {
        // nothing to do
    }

    int Node::getNumParameters() const
    {
        return 0;
    }

    double Node::getError(double targetValue, bool derivative) const
    {
        const double diff = m_value - targetValue;
        if (derivative)
        {
            return 2 * diff;
        }

        return diff * diff;
    }

    // --------------------------
    // InnerNode
    // --------------------------
    InnerNode::InnerNode(const std::vector<std::shared_ptr<Edge>>& inputEdges)
        : m_inputEdges(inputEdges)
        , m_bias(0.f)
    {
    }

    InnerNode::~InnerNode()
    {
        m_inputEdges.clear();
    }

    bool InnerNode::assignParameters(std::queue<double>& params)
    {
        if (static_cast<int>(params.size()) < getNumParameters())
        {
            return false;
        }

        for (auto& edge : m_inputEdges)
        {
            edge->assignParameters(params);
        }

        m_bias = params.front();
        params.pop();

        return true;
    }

    void InnerNode::getParameters(std::vector<double>& params) const
    {
        for (auto& edge : m_inputEdges)
        {
            edge->getParameters(params);
        }

        params.push_back(m_bias);
    }

    int InnerNode::getNumParameters() const
    {
        return static_cast<int>(m_inputEdges.size()) + 1;
    }

    void InnerNode::updateValue(std::function<double(double, bool)> activationFunction)
    {
        m_value = activationFunction(calculateBaseValue(), false);
    }

    double InnerNode::calculateBaseValue() const
    {
        double value = m_bias;

        // sum of all incoming edge values
        for (const auto& ie : m_inputEdges)
        {
            value += ie->getValue();
        }

        return value;
    }

    void InnerNode::handleBackpropagation(double targetValue, std::function<double(double, bool)> activationFunction, std::vector<double>& inputValueAdjustments)
    {
        const double nonActivatedValue = calculateBaseValue();
        const double derivedActivationValue = activationFunction(nonActivatedValue, true);

        const double derivedError = getError(targetValue, true);
        const double multiplier = derivedActivationValue * derivedError;

        m_bias = multiplier;

        if (inputValueAdjustments.empty())
        {
            inputValueAdjustments.resize(m_inputEdges.size());
        }

        assert(inputValueAdjustments.size() == m_inputEdges.size());

        for (int k = 0; k < m_inputEdges.size(); k++)
        {
            inputValueAdjustments[k] += m_inputEdges[k]->handleBackpropagation(multiplier);
        }
    }
}
