#pragma once

#include <vector>

#include "General/Globals.h"

namespace Game
{
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
        virtual void getExpectedOutput(int playerId, std::vector<double>& expectedOutput) const = 0;
        /// correct output values to be closer to what was expected
        virtual void correctOutputValues(int playerId, std::vector<double>& outputValues) const = 0;
        virtual GameState evaluateBoard() const = 0;

    public:
        int getBoardSize() const;
        int countCellState(const CellState state) const;
        CellState getCellValue(int row, int col) const;
        virtual void setGameCells(const std::vector<CellState>& gameCells);
        virtual void getGameCells(std::vector<CellState>& gameCells) const;

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
        void getExpectedOutput(int playerId, std::vector<double>& expectedOutput) const override;
        void correctOutputValues(int playerId, std::vector<double>& outputValues) const override;
        GameState evaluateBoard() const override;

    public:
        static void capValueAccordingToState(const CellState& state, double& value);
        static void getTripleCandidates(const std::vector<CellState>& gameCells, CellState targetState, std::vector<int>& candidates);
        static int getTripleCandidate(const std::vector<CellState>& gameCells, CellState targeState, int idx1, int idx2, int idx3);

        /// collect all possible permutations of values the game board can have at the last turn
        /// (when only one cell remains empty and the game hasn't been decided yet)
        static void collectInconclusiveFinalGameBoardStates(std::vector<std::vector<CellState>>& collectedGameStates);
    };
}
