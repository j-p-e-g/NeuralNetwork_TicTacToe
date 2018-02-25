#pragma once

#include <vector>

#include "General/Globals.h"

namespace Game
{
    enum GameState
    {
        GS_ONGOING,
        GS_GAMEOVER_WON,
        GS_GAMEOVER_LOST,
        GS_GAMEOVER_TIMEOUT
    };

    enum CellState
    {
        CS_EMPTY = 0,
        CS_PLAYER1 = 1,
        CS_PLAYER2 = -1
    };

    class GameLogic
    {
    public:
        GameLogic() = default;
        ~GameLogic() = default;

    public:
        static std::string getGameStateDescription(GameState state);

    public:
        virtual void initBoard() = 0;
        virtual void getRequiredNetworkSize(NetworkSizeData& sizeData) const = 0;
        virtual bool isValidMove(int playerId, int actionIndex) const = 0;
        virtual bool applyMove(int playerId, int actionIndex) = 0;
        virtual void getNodeNetworkInputValues(std::vector<double>& inputValues) const = 0;
        virtual GameState evaluateBoard() const = 0;

    public:
        int getBoardSize() const;
        int countCellState(const CellState state) const;
        CellState getCellValue(int row, int col) const;

    protected:
        std::vector<CellState> m_gameCells;
        int m_numRows = 0;
        int m_numCols = 0;
    };

    class TicTacToeLogic
        : public GameLogic
    {
    public:
        TicTacToeLogic();
        ~TicTacToeLogic() = default;

    public:
        void initBoard() override;
        void getRequiredNetworkSize(NetworkSizeData& sizeData) const override;
        bool isValidMove(int playerId, int actionIndex) const override;
        bool applyMove(int playerId, int actionIndex) override;
        void getNodeNetworkInputValues(std::vector<double>& inputValues) const override;
        GameState evaluateBoard() const override;
    };
}