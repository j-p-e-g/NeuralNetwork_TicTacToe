#include "stdafx.h"

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
        case GS_GAMEOVER_WON: return "won";
        case GS_GAMEOVER_LOST: return "lost";
        case GS_GAMEOVER_TIMEOUT: return "timeout";
        case GS_ONGOING: return "in progress";
        default: return "";
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
        sizeData.numHiddenNodes = std::vector<int>();
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
}
