#include "stdafx.h"

#include <algorithm>
#include <assert.h>

#include "GameLogic.h"

namespace Game
{
    CellState GameLogic::getCellValue(int row, int col) const
    {
        assert(row >= 0 && row < m_numRows);
        assert(col >= 0 && col < m_numCols);

        const int index = row * m_numCols + col;
        assert(index >= 0 && index < getBoardSize());

        return m_gameCells[index];
    }

    void GameLogic::setGameCells(const std::vector<CellState>& gameCells)
    {
        assert(gameCells.size() == getBoardSize());
        m_gameCells = gameCells;
    }

    void GameLogic::getGameCells(std::vector<CellState>& gameCells) const
    {
        gameCells = m_gameCells;
    }

    int GameLogic::getBoardSize() const
    {
        return m_numRows * m_numCols;
    }

    int GameLogic::countCellState(const CellState state) const
    {
        int count = 0;
        for (const auto& cell : m_gameCells)
        {
            if (cell == state)
            {
                count++;
            }
        }

        return count;
    }

    std::string GameLogic::getGameStateDescription(GameState state)
    {
        switch (state)
        {
        case GS_GAMEOVER_WON:     return "won";
        case GS_GAMEOVER_LOST:    return "lost";
        case GS_GAMEOVER_TIMEOUT: return "timeout";
        case GS_ONGOING:          return "in progress";
        case GS_INVALID:          return "invalid";
        default:                  return "";
        }
    }

    //-------------------------------
    // Tic Tac Toe
    //-------------------------------
    TicTacToeLogic::TicTacToeLogic()
    {
        m_numRows = 3;
        m_numCols = 3;

        initBoard();
    }

    void TicTacToeLogic::initBoard()
    {
        m_gameCells.clear();
        m_gameCells.reserve(getBoardSize());

        for (int k = 0; k < getBoardSize(); k++)
        {
            m_gameCells.push_back(CellState::CS_EMPTY);
        }
    }

    void TicTacToeLogic::getRequiredNetworkSize(NetworkSizeData& sizeData) const
    {
        sizeData.numOutputNodes = getBoardSize();
        sizeData.numInputNodes = 2 * sizeData.numOutputNodes;
    }

    bool TicTacToeLogic::isValidMove(int playerId, int actionIndex) const
    {
        if (playerId < 0 || playerId > 1)
        {
            return false;
        }

        if (actionIndex < 0 || actionIndex >= static_cast<int>(m_gameCells.size()))
        {
            return false;
        }

        return (m_gameCells[actionIndex] == CS_EMPTY);
    }

    bool TicTacToeLogic::applyMove(int playerId, int actionIndex)
    {
        if (!isValidMove(playerId, actionIndex))
        {
            return false;
        }

        m_gameCells[actionIndex] = (playerId == 0 ? CS_PLAYER1 : CS_PLAYER2);
        return true;
    }

    void TicTacToeLogic::getNodeNetworkInputValues(std::vector<double>& inputValues) const
    {
        inputValues.clear();

        for (const auto& cell : m_gameCells)
        {
            switch (cell)
            {
            case CellState::CS_EMPTY:
                inputValues.push_back(-1.0);
                inputValues.push_back(-1.0);
                break;
            case CellState::CS_PLAYER1:
                inputValues.push_back(1.0);
                inputValues.push_back(-1.0);
                break;
            case CellState::CS_PLAYER2:
                inputValues.push_back(-1.0);
                inputValues.push_back(1.0);
                break;
            }
        }
    }

    void TicTacToeLogic::getExpectedOutput(int playerId, std::vector<double>& expectedOutput) const
    {
        const double OCCUPIED_CELL_VALUE = 0;
        const double WIN_VALUE = 1;
        const double LOSS_PREVENTION_VALUE = 1;
        const double EMPTY_CELL_VALUE_WITH_TRIPLES = 0.5;
        const double EMPTY_CELL_VALUE_NO_TRIPLES = 1;

        std::vector<int> winTripleCandidates;
        std::vector<int> lossTripleCandidates;

        getTripleCandidates(m_gameCells, playerId == 0 ? CellState::CS_PLAYER1 : CellState::CS_PLAYER2, winTripleCandidates);
        getTripleCandidates(m_gameCells, playerId == 0 ? CellState::CS_PLAYER2 : CellState::CS_PLAYER1, lossTripleCandidates);

        expectedOutput.clear();
        for (int k = 0; k < getBoardSize(); k++)
        {
            if (m_gameCells[k] != CellState::CS_EMPTY)
            {
                expectedOutput.push_back(OCCUPIED_CELL_VALUE);
                continue;
            }

            if (std::find(winTripleCandidates.begin(), winTripleCandidates.end(), k) != winTripleCandidates.end())
            {
                expectedOutput.push_back(WIN_VALUE);
            }
            else if (std::find(lossTripleCandidates.begin(), lossTripleCandidates.end(), k) != lossTripleCandidates.end())
            {
                expectedOutput.push_back(LOSS_PREVENTION_VALUE);
            }
            else if (winTripleCandidates.empty() && lossTripleCandidates.empty())
            {
                expectedOutput.push_back(EMPTY_CELL_VALUE_NO_TRIPLES);
            }
            else
            {
                expectedOutput.push_back(EMPTY_CELL_VALUE_WITH_TRIPLES);
            }
        }
    }

    void TicTacToeLogic::correctOutputValues(int playerId, std::vector<double>& outputValues) const
    {
        assert(outputValues.size() == m_gameCells.size());

        for (unsigned int k = 0; k < m_gameCells.size(); k++)
        {
            capValueAccordingToState(m_gameCells[k], outputValues[k]);
        }
    }

    void TicTacToeLogic::capValueAccordingToState(const CellState& state, double& value)
    {
        switch (state)
        {
        case CS_EMPTY:
            if (value < 1)
            {
                value = 1;
            }
            break;
        default:
            if (value > 0)
            {
                value = 0;
            }
            break;
        }
    }

    void TicTacToeLogic::getTripleCandidates(const std::vector<CellState>& gameCells, CellState targeState, std::vector<int>& candidates)
    {
        // check if we can complete a triple
        for (int k = 0; k < 3; k++)
        {
            // compare rows
            int candidate = getTripleCandidate(gameCells, targeState, 3 * k, 3 * k + 1, 3 * k + 2);
            if (candidate != -1)
            {
                candidates.push_back(candidate);
            }

            // compare columns
            candidate = getTripleCandidate(gameCells, targeState, k, 3 + k, 6 + k);
            if (candidate != -1)
            {
                candidates.push_back(candidate);
            }
        }

        // compare diagonals
        int candidate = getTripleCandidate(gameCells, targeState, 0, 4, 8);
        if (candidate != -1)
        {
            candidates.push_back(candidate);
        }

        candidate = getTripleCandidate(gameCells, targeState, 2, 4, 6);
        if (candidate != -1)
        {
            candidates.push_back(candidate);
        }
    }

    int TicTacToeLogic::getTripleCandidate(const std::vector<CellState>& gameCells, CellState targetState, int idx1, int idx2, int idx3)
    {
        assert(idx1 >= 0 && idx1 < gameCells.size());
        assert(idx2 >= 0 && idx2 < gameCells.size());
        assert(idx3 >= 0 && idx3 < gameCells.size());

        const CellState cs1 = gameCells[idx1];
        const CellState cs2 = gameCells[idx2];
        const CellState cs3 = gameCells[idx3];

        if (cs1 == CS_EMPTY && cs2 == targetState && cs2 == cs3)
        {
            return idx1;
        }

        if (cs2 == CS_EMPTY && cs1 == targetState && cs1 == cs3)
        {
            return idx2;
        }

        if (cs3 == CS_EMPTY && cs1 == targetState && cs1 == cs2)
        {
            return idx3;
        }

        return -1;
    }

    GameState TicTacToeLogic::evaluateBoard() const
    {
        for (int k = 0; k < 3; k++)
        {
            // compare cells within a row
            CellState firstCellState = getCellValue(k, 0);
            if (firstCellState != CellState::CS_EMPTY && firstCellState == getCellValue(k, 1) && firstCellState == getCellValue(k, 2))
            {
                return GameState::GS_GAMEOVER_WON;
            }

            // compare cells within a column
            firstCellState = getCellValue(0, k);
            if (firstCellState != CellState::CS_EMPTY && firstCellState == getCellValue(1, k) && firstCellState == getCellValue(2, k))
            {
                return GameState::GS_GAMEOVER_WON;
            }
        }

        // compare diagonals
        const CellState centerState = getCellValue(1, 1);
        if (centerState != CS_EMPTY)
        {
            if (centerState == getCellValue(0, 0) && centerState == getCellValue(2, 2)
                || centerState == getCellValue(0, 2) && centerState == getCellValue(2, 0))
            {
                return GameState::GS_GAMEOVER_WON;
            }
        }

        if (countCellState(CellState::CS_EMPTY) == 0)
        {
            // nobody won, but we're out of turns
            return GS_GAMEOVER_TIMEOUT;
        }

        // otherwise the game is still on
        return GameState::GS_ONGOING;
    }

    void TicTacToeLogic::collectInconclusiveFinalGameBoardStates(std::vector<std::vector<CellState>>& collectedGameStates)
    {
        collectedGameStates.clear();

        // last turn: each player has made 4 moves, and one cell is empty
        std::vector<CellState> cellStates;
        cellStates.push_back(CellState::CS_EMPTY);
        for (int k = 0; k < 4; k++)
        {
            cellStates.push_back(CellState::CS_PLAYER1);
        }
        for (int k = 0; k < 4; k++)
        {
            cellStates.push_back(CellState::CS_PLAYER2);
        }

        // next_permutation requires the vector to be sorted
        std::sort(cellStates.begin(), cellStates.end());

        TicTacToeLogic logic;
        do
        {
            // for each permutation, prepare the board accordingly
            logic.initBoard();
            for (unsigned int cellId = 0; cellId < cellStates.size(); cellId++)
            {
                const CellState state = cellStates[cellId];
                if (state != CellState::CS_EMPTY)
                {
                    logic.applyMove(state == CellState::CS_PLAYER1 ? 0 : 1, cellId);
                }
            }

            // ... but only add the permutation if neither player has won already
            const GameState gameState = logic.evaluateBoard();
            if (gameState == GS_ONGOING)
            {
                collectedGameStates.push_back(cellStates);
            }
        }
        while (std::next_permutation(cellStates.begin(), cellStates.end()));
    }
}
