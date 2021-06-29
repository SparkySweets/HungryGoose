import torch
import copy
import random
from NNet import GooseNet


class DQN():
    # Заменил параметры s во всех методах на states, y -> y_exp
    # вынес все unsqueeze() во вне, чтобы класс работал с теми же данными, как nn.Module (с батчами)
    def __init__(self, n_channels, lr=1e-5):
        self.criterion = torch.nn.MSELoss()
        self.model = GooseNet(n_channels)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.model_target = copy.deepcopy(self.model)
    
    def update(self, states, y_exp):
        """
        Обновляет веса DQN, получив обучающий пример
        @param states: состояния
        @param y_exp: целевые значения
        """
        y_pred = self.model(states)
        loss = self.criterion(y_pred, y_exp)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def predict(self, states):
        """
        Вычисляет значения Q-функции состояния для всех действий, применяя обученную модель
        @param states: входные состояния
        @return: значения Q для всех действий
        """
        with torch.no_grad():
            return self.model(states)
        
    def target_predict(self, states):
        """
        Вычисляет значения Q-функции состояния для всех действий, с помощью целевой сети
        @param states: входные состояния
        @return: значения Q для всех действий
        """
        with torch.no_grad():
            return self.model(states)
    
    # метод для синхронизации весов
    def copy_target(self):
        self.model_target.load_state_dict(self.model.state_dict())
        
    def replay(self, memory, replay_size, gamma):
        # метод работает с состояними, тут уже unsqueeze приходится делать внутри
        # также осуществляю перевод в тензоры списка states и td_targets с помощью torch.stack 
        # (вызов конструткора Tensor(лист тензоров) не работает, так как он тока для листов скаляров, на тензор ругается)
        # для перевода в тензор td_targets подходил и просто конструктор, т.к. q_values был листом скаляров (тензор был преобразован методом tolist()),
        # но это я убрал для единообразия
        """
        буфер воспроизведения совместно с целевой сетью
        @param memory: a list of experience
        @param replay_size: the number of samples we use to update the model each time
        @param gamma: the discount factor
        """
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)
            states = []
            td_targets = []
            for state, action, next_state, reward, is_done in replay_data:
                #print(is_done)
                #print(state.shape)
                #print(len(state.tolist()[0]))
                states.append(state)
                #print(state.tolist())
                q_values = self.predict(state.unsqueeze(0)).squeeze()
                #print(self.predict(state).tolist())
                #print(q_values)
                if is_done:
                    q_values[action] = reward
                else:
                    q_values_next = self.target_predict(next_state.unsqueeze(0)).detach()
                    #print(q_values_next)
                    #print(torch.max(q_values_next))
                    #q_values[action] = reward + gamma * torch.max(q_values_next).item()
                    #print(q_values)
                    q_values[action] = reward + gamma * torch.max(q_values_next)


                td_targets.append(q_values)

            self.update(torch.stack(states), torch.stack(td_targets))