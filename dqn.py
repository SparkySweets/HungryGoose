import torch
import copy
import random
from NNet import GooseNet


class DQN():
    # Заменил параметры s во всех методах на states, y -> y_exp
    # вынес все unsqueeze() во вне, чтобы класс работал с теми же данными, как nn.Module (с батчами)
    def __init__(self, n_channels, lr=1e-5):
        self.criterion = torch.nn.MSELoss()
        self.model = GooseNet(n_channels).to('cpu')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.model_target = copy.deepcopy(self.model)
        self.model_target.eval()

    def update(self, y_pred, y_exp):
        # тут вызывалась self.model(), хотя в replay мы уже делали predict()
        # так же loss считался не для батча из значений Q для выбранного нами действия, а для батча
        # всех действий. Это делать незачем, так как изменения произошли только для Q соотвестующего
        # выбранному action
        """
        Обновляет веса DQN, получив обучающий пример
        @param states: состояния
        @param y_exp: целевые значения
        """
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
        self.model.eval()
        with torch.no_grad():
            return self.model(states)

    def target_predict(self, states):
        """
        Вычисляет значения Q-функции состояния для всех действий, с помощью целевой сети
        @param states: входные состояния
        @return: значения Q для всех действий
        """
        self.model_target.eval()
        with torch.no_grad():
            return self.model_target(states)

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
        if len(memory) < replay_size:
            return
        self.model.train()
        self.model_target.eval()

        replay_data = random.sample(memory, replay_size)

        state_batch = torch.stack(list(map(lambda x: x[0], replay_data)))
        action_batch = torch.tensor(list(map(lambda x: x[1], replay_data)))
        next_state_batch = torch.stack(list(map(lambda x: x[2], replay_data)))
        reward_batch = torch.tensor(list(map(lambda x: x[3], replay_data)), dtype=torch.float)
        is_not_done_batch = torch.tensor(list(map(lambda x: x[4] == False, replay_data)),  dtype=torch.bool)

        y_pred = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).unsqueeze(1)
        # ожидаемая награда = многовенная нагрда + ... (зависит от того, терминальное ли состояние)
        y_exp = reward_batch
        # добавим ожидаемое значение Q, при условии, что мы попали в нетерминальное состояние
        non_terminal_next_states = next_state_batch[is_not_done_batch]
        next_state_q_values = self.model_target(non_terminal_next_states).max(1)[0].detach()
        y_exp[is_not_done_batch] = y_exp[is_not_done_batch] + gamma * next_state_q_values

        self.update(y_pred, y_exp)
