#include "stdafx.h"

#include <memory>

#include "Game/GameLogic.h"
#include "Training/TicTacToeTrainer.h"

int main()
{
    std::shared_ptr<Game::GameLogic> gameLogic = std::make_shared<Game::TicTacToeLogic>();
    Training::TicTacToeTrainer trainer(gameLogic);
    trainer.run();
    return 0;
}
