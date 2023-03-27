import copy
import math
import random
import sys

import numpy as np

N = 20  # 工件个数#
random.seed(1)
job_normal_time = [random.randint(1, 16) for _ in range(N)]  # 工件的标准实际加工时间#
job_size = [random.randint(1, 15) for _ in range(N)]  # 工件的尺寸#
random.seed()
C = 40  # 机器的容量#
alpha = 0.25  # 成本系数#
beta = 0.55
gamma = 0.05
delta = 0.15
b = 0.01  # 工件加工时间衰退系数#
tau = -0.1  # 工件加工时间学习系数#
mu = -0.1  # 维修活动的学习系数#
t_0 = 10  # 维修活动标准加工时间#
inta = 1  # 用于计算交叉概率#


# 用于生成不同的加工时间和尺寸#
def job_example_set(N_ex, time_low, time_upper, size_low, size_upper, seed):
    global N
    N = N_ex
    random.seed(seed)
    global job_normal_time
    job_normal_time = [random.randint(time_low, time_upper) for _ in range(N)]
    global job_size
    job_size = [random.randint(size_low, size_upper) for _ in range(N)]
    random.seed()


# k = int(N * (delta - gamma) / alpha)
# m = int(N * (beta - delta) / beta) - k

# 按照编码组批#
def batch_set(code):
    """组批"""
    x = copy.deepcopy(code)
    x = [-value for value in x]
    x = np.argsort(x)
    length = len(x)
    batch = [[] for _ in range(length)]  # 每个batch存放工件的索引#
    batch_rest_memory = [C for _ in range(length)]
    batch_time = [0 for _ in range(length)]  # batch的加工时间#
    for i in range(length):
        normal_time = job_normal_time[x[i]]
        size = job_size[x[i]]
        for j in range(length):
            if batch_rest_memory[j] >= size:
                batch[j].append(x[i])
                batch_time[j] = max(normal_time, batch_time[j])
                batch_rest_memory[j] -= size
                break
    """计算batch 个数n"""
    n = 0
    for time in batch_time:
        if time != 0:
            n += 1
        else:
            break
    batch = batch[0:n]
    batch_time = batch_time[0:n]
    return batch, batch_time


def cal_min_cost(global_batch, global_batch_time):
    n = len(global_batch)
    """algorithm 1 求解"""
    k = int(n * (delta - gamma) / alpha)
    k_m = int(n * (beta - delta) / beta)
    total_cost = []
    batch_time = copy.deepcopy(global_batch_time)
    batch_time = sorted(batch_time)
    for e in range(1, n + 1):
        cost = 0
        index_time = [0 for _ in range(len(batch_time))]

        if e < k:
            M = (alpha * e + n * gamma) * t_0 * math.pow(e, mu)
        elif e < k_m:
            M = n * delta * t_0 * math.pow(e, mu)
        else:
            M = beta * (n - e) * t_0 * math.pow(e, mu)
        w_i = []
        for i in range(1, n + 1):
            if i <= k:
                w_i.append(alpha * (i - 1) + n * gamma)
            elif i <= k_m:
                w_i.append(n * delta)
            elif i <= n:
                w_i.append(beta * (n - i + 1))
            else:
                print('fault1')
        W_i = []

        for i in range(1, n + 1):
            if i <= e:
                value = 0
                value += w_i[i - 1]
                for y in range(i + 1, e + 1):
                    value += (b * math.pow((1 + b), y - i - 1) * w_i[y - 1] * math.pow(y, tau))
                W_i.append(value)
            elif i <= n - 1:
                value = 0
                value += w_i[i - 1]
                for y in range(i + 1, n + 1):
                    # 这里的w_i[y-1]对应伪代码的w_y#
                    value += (b * math.pow((1 + b), y - i - 1) * w_i[y - 1] * math.pow(y - e, tau))
                W_i.append(value)
            elif i == n:
                value = 0
                value += w_i[i - 1]
                W_i.append(value)
            else:
                print('fault2')
        W_i_copy = copy.deepcopy(W_i)
        for it in range(1, n + 1):
            max_index = np.argmax(W_i)
            W_i[max_index] = 0
            index_time[max_index] = batch_time[it - 1]
        for i in range(n):
            cost += W_i_copy[i] * index_time[i]
        cost += M
        total_cost.append(cost)
    return np.argmin(total_cost) + 1, min(total_cost)


def decode(code):
    batch, batch_time = batch_set(code)
    return cal_min_cost(batch, batch_time)[1]


def encode(batch):
    length = 0
    for every in batch:
        length += len(every)
    x = copy.deepcopy(batch)
    start = 1
    step = 1 / length
    code = [0 for _ in range(length)]
    for every_batch in x:
        for index in every_batch:
            code[index] = start
            start -= step
    return code


# GSO_VNS交叉算子#
def crossover(pop1, pop2, p):
    x = copy.deepcopy(pop1)
    y = copy.deepcopy(pop2)
    x_new = copy.deepcopy(x)
    y_new = copy.deepcopy(y)
    for i in range(len(x)):
        if random.random() > p:
            x_new[i] = y[i]
            y_new[i] = x[i]
    pop = [x, y, x_new, y_new]
    fit = []
    for code in pop:
        fit.append(decode(code))
    index1 = np.argmin(fit)

    return pop[index1]


# 局部搜索策略#
def local_search_strategy(batch, batch_time):
    current_batch = copy.deepcopy(batch)
    current_batch_time = copy.deepcopy(batch_time)
    while True:
        length = len(batch)
        index1 = random.randint(0, length - 2)
        index2 = index1 + 1
        batch1 = copy.deepcopy(current_batch[index1])
        batch2 = copy.deepcopy(current_batch[index2])
        batch1.extend(batch2)
        batch_merge = copy.deepcopy(batch1)  # batch_batch2混合#
        batch_new = []  # 按照工件加工时间重新排序#
        time = []
        for index in batch_merge:
            time.append(job_normal_time[index])
        for i in range(len(time)):
            index = np.argmax(time)
            batch_new.append(batch_merge[index])
            time[index] = 0
        result_batch = [C, C]
        batch1 = []  # 存放重新生成的两个batch#
        batch2 = []
        time = [0, 0]
        for index in batch_new:  # 根据容量判断重新组批是否能生成两个batch,不能则直接返回上一个组合#
            if result_batch[0] >= job_size[index]:
                result_batch[0] -= job_size[index]
                batch1.append(index)
            elif result_batch[1] >= job_size[index]:
                result_batch[1] -= job_size[index]
                batch2.append(index)
            else:
                # 这里必然会返回#
                return current_batch, current_batch_time
        if batch1 == current_batch[index1] and batch2 == current_batch[index2]:
            for i in range(len(current_batch_time)):
                if current_batch_time[i] == 0:
                    del current_batch_time[i]
                    del current_batch[i]
            return current_batch, current_batch_time
        for index in batch1:
            time[0] = max(time[0], job_normal_time[index])
        for index in batch2:
            time[1] = max(time[1], job_normal_time[index])
        current_batch[index1] = copy.deepcopy(batch1)
        current_batch[index2] = copy.deepcopy(batch2)
        current_batch_time[index1] = time[0]
        current_batch_time[index2] = time[1]


# 计算前进方向#
def cal_di(angle_list):
    di = []
    head_angle = copy.deepcopy(angle_list)
    for j in range(len(head_angle)):
        temp = 1
        if j == 0:
            for angle in head_angle:
                temp *= math.cos(angle)
            di.append(temp)
        else:
            temp *= math.sin(head_angle[j - 1])
            for q in range(j, len(head_angle)):
                angle = head_angle[q]
                temp *= math.cos(angle)
            di.append(temp)
    temp = 1
    temp *= math.sin(head_angle[-1])
    di.append(temp)
    return di


def GSO_VNS():
    result = []
    t_max = 400  # 迭代次数#
    pop_long = 10  # 种群个数#
    I = round(math.sqrt(N + 1))  # GSO算法参数#
    v_max = round(2 * math.sqrt(N))
    theta_max = math.pi / math.pow(I, 2)
    h_max = theta_max / 2
    pr_list = [0.9, 0.8, 0.7, 0.6]  # 游荡者比率#

    se = 0
    itt = 0  # 当前迭代代数
    nu = 0

    population = np.random.random(size=(pop_long, N))  # 种群#
    fitness = [decode(population[i]) for i in range(len(population))]  # 种群适应度#
    best_pop_index = np.argmin(fitness)
    best_fitness = fitness[best_pop_index]  # 最优个体对应的适应度#
    best_pop = copy.deepcopy(population[best_pop_index])  # 最优个体#
    fit_o = sys.maxsize
    producer = None  # 发现者#
    head_angle = np.random.random(size=(pop_long, N - 1)) * 2 * math.pi  # 初始的head—angle从在0-2pi中随机生成#
    head_angle_history = [copy.deepcopy(head_angle)]

    best_batch = None
    best_batch_time = None

    while itt < t_max:
        if fit_o >= best_fitness:
            se = 0
            producer = copy.deepcopy(best_pop)
        else:
            p = math.pow(math.e, -itt) / (inta + math.pow(math.e, -itt))
            producer = copy.deepcopy(crossover(producer, best_pop, p))
        fit_o = min(fit_o, decode(producer))

        r1 = random.random()
        r2 = np.random.random(size=(N - 1,))

        angle_mid = copy.deepcopy(head_angle[best_pop_index])
        di = cal_di(angle_mid)
        di = np.array(di)
        x_new1 = (best_pop + r1 * v_max * di)

        angle_right = np.array(copy.deepcopy(angle_mid))
        angle_right += (r2 * theta_max / 2)
        di_right = cal_di(angle_right)
        di_right = np.array(di_right)
        x_new2 = (best_pop + r1 * v_max * di_right)

        angle_left = np.array(copy.deepcopy(angle_mid))
        angle_left -= (r2 * theta_max / 2)
        di_left = cal_di(angle_left)
        di_left = np.array(di_left)
        x_new3 = (best_pop + r1 * v_max * di_left)

        x = [x_new1, x_new2, x_new3]
        y = [decode(code) for code in x]
        r2 = np.array([r2 for _ in range(pop_long)])
        if fit_o > min(y):
            fit_o = min(y)
            index = np.argmin(y)
            producer = x[index]
            head_angle += h_max * r2
            nu = 0
        else:
            nu += 1
            if nu > I:
                # 更换成I次之前的角度#
                head_angle = head_angle_history[itt - I]
                nu = 0

        # 跟随者#
        number = pop_long * (1 - pr_list[se])
        scrounger = random.sample(range(pop_long), int(number))
        r3 = np.random.random(size=(N,))
        for index in scrounger:
            pop = copy.deepcopy(population[index])
            producer = np.array(producer)
            population[index] = (pop + r3 * (producer - pop))

        # 游荡者#
        ranger = [value for value in range(pop_long) if value not in scrounger]
        for index in ranger:
            Vg = I * r1 * v_max
            pop = copy.deepcopy(population[index])
            di_ranger = cal_di(head_angle[index])
            di_ranger = np.array(di_ranger)
            population[index] = copy.deepcopy(pop + Vg * di_ranger)
            head_angle[index] = copy.deepcopy(head_angle[index] + r2[index] * h_max)

        head_angle_history.append(copy.deepcopy(head_angle))

        fit = [decode(population[i]) for i in range(pop_long)]
        best_fitness = min(fit)
        best_pop_index = np.argmin(fit)
        best_pop = copy.deepcopy(population[best_pop_index])

        se += 1
        if se > 3:
            se = 0

        # local-search#
        code_batch, code_batch_time = batch_set(producer)
        best_batch, best_batch_time = local_search_strategy(code_batch, code_batch_time)
        local_search_result = cal_min_cost(best_batch, best_batch_time)[1]
        if local_search_result < fit_o:
            producer = encode(best_batch)
            fit_o = local_search_result

        print('第{}代： ,编码：{} ,fitness:{}'.format(itt, producer, fit_o))
        # print('producer batch:{}, batch_time:{}, fit_o:{}:'.format(best_batch, best_batch_time, fit_o))
        # print('producer batch:{}, batch_time:{}, fit_o:{}:'.format(best_batch, best_batch_time,
        #                                                            cal_min_cost(best_batch, best_batch_time)[1]))
        print('local search batch:{}, batch_time:{},local-search_fit:{}'.format(best_batch, best_batch_time,
                                                                                local_search_result))
        result.append(copy.deepcopy(fit_o))
        itt += 1
    return result


# 计算下界#
def cal_lower_bound(normal_time, normal_size):
    x = copy.deepcopy(normal_time)
    y = copy.deepcopy(normal_size)
    min_size = min(y)
    length = len(y)
    x = [-value for value in x]
    x = np.argsort(x)

    batch = [[] for _ in range(length)]
    batch_rest_memory = [C for _ in range(length)]
    batch_time = [0 for _ in range(length)]
    for i in range(length):
        time = normal_time[x[i]]
        size = min_size
        for j in range(length):
            if batch_rest_memory[j] >= size:
                batch[j].append(x[i])
                batch_time[j] = max(time, batch_time[j])
                batch_rest_memory[j] -= size
                break

    n = 0
    for time in batch_time:
        if time != 0:
            n += 1
        else:
            break
    batch = batch[0:n]
    batch_time = batch_time[0:n]
    result = cal_min_cost(batch, batch_time)[1]
    # print('下界，batch:{} , batch_time:{} ,value:{} '.format(batch, batch_time, result))
    return result


# 计算RPD#
def CAL_RPD(num):
    fit_best = min(num)
    fit_alg = sum(num) / len(num)
    return 100 * (fit_alg - fit_best) / fit_best


# 预实验获取inta#
def preliminary_test(count):
    N_list = [20, 50, 100, 150]
    inta_list = [1, 1.5, 2, 2.5, 3, 3.5]
    result = np.zeros(shape=[count * len(N_list), len(inta_list)])
    for i in range(len(N_list)):
        for j in range(count):
            fit_list = []
            rpd_list = []
            job_example_set(N_ex=N_list[i], time_low=1,
                            time_upper=15, size_low=1, size_upper=15, seed=j)
            print(f'N_job:{N_list[i]},count:{j}')
            for k in range(len(inta_list)):
                global inta
                inta = inta_list[k]
                fit = min(GSO_VNS())
                fit_list.append(fit)
            for value in fit_list:
                rpd = (value - min(fit_list)) / min(fit_list)
                rpd_list.append(rpd)
            result[i * count + j] = copy.deepcopy(np.array(rpd_list))

    np.savetxt('D:\\computer\\PythonCode\\陆老师代码\\预实验.txt', X=result, fmt='%.2f')

    return result


# ---------------------------------------GA-----------------------------------------------#
# 初始化种群#
def init_population(pop_size, gene_length):
    population = []
    chromosome = []
    for i in range(gene_length):
        chromosome.append(i)
    for i in range(pop_size):
        random.shuffle(chromosome)
        population.append(copy.deepcopy(chromosome))
    time = job_normal_time[:]
    time = [-value for value in job_normal_time]

    chromosome = np.argsort(time).tolist()
    population[0] = copy.deepcopy(chromosome)
    return population


# 计算适应度函数值
def fitness_function(chromosome):
    batch, batch_time = batch_set_GA(chromosome)
    result = cal_min_cost(batch, batch_time)[1]
    return result


# 二元锦标赛选择
def tournament_selection(population):
    parent1 = random.choice(population)
    parent2 = random.choice(population)
    count = 0
    while parent2 == parent1:
        if count > 10:
            break
        count += 1
        parent2 = random.choice(population)
    if fitness_function(parent1) < fitness_function(parent2):
        return parent1
    else:
        return parent2


# 单点交叉
def single_point_crossover(parent1, parent2):
    point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:point]
    child2 = parent2[:point]
    # 重新排列
    x = parent1[point:]
    y = parent2[point:]
    for i in range(len(parent1)):
        if parent1[i] in y:
            child2.append(parent1[i])
        if parent2[i] in x:
            child1.append(parent2[i])
    return child1, child2


# 单点变异
def single_point_mutation(chromosome):
    for _ in range(10):
        point = random.randint(0, len(chromosome) - 2)
        chromosome[point], chromosome[point + 1] = chromosome[point + 1], chromosome[point]
    return chromosome


def batch_set_GA(code):
    x = copy.deepcopy(code)
    length = len(job_normal_time)
    batch = [[] for _ in range(length)]  # 每个batch存放工件的索引#
    batch_rest_memory = [C for _ in range(length)]
    batch_time = [0 for _ in range(length)]  # batch的加工时间#
    for i in range(length):
        normal_time = job_normal_time[x[i]]
        size = job_size[x[i]]
        for j in range(length):
            if batch_rest_memory[j] >= size:
                batch[j].append(x[i])
                batch_time[j] = max(normal_time, batch_time[j])
                batch_rest_memory[j] -= size
                break
    """计算batch 个数n"""
    n = 0
    for time in batch_time:
        if time != 0:
            n += 1
        else:
            break
    batch = batch[0:n]
    batch_time = batch_time[0:n]
    return batch, batch_time


# GA算法
def GA(population_size, gene_length, generation):
    population = init_population(population_size, gene_length)
    for i in range(generation):
        population_copy = []
        print(f'第{i + 1}代')
        for j in range(len(population)):
            # 选择
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            # 交叉
            child1, child2 = single_point_crossover(parent1, parent2)
            # 变异
            child1 = single_point_mutation(child1)
            child2 = single_point_mutation(child2)
            pop_list = [parent1, parent2, child1, child2]
            fit_list = [fitness_function(pop_list[i]) for i in range(len(pop_list))]
            pop = pop_list[np.argmin(fit_list)]
            # pop = min(pop_list, key=fitness_function)
            population_copy.append(pop[:])
        population = population_copy
        res = min(population, key=fitness_function)
        print(res)
        print(fitness_function(res))
    return min(population, key=fitness_function)


if __name__ == '__main__':
    # res = GSO_VNS()
    # lower_bound = cal_lower_bound(job_normal_time, job_size)
    # preliminary_test(25)
    # GSO_VNS()
    job_example_set(20, 1, 16, 1, 15, 3)
    print(cal_lower_bound(job_normal_time, job_size))
    GSO_VNS()
    GA(10, N, 400)
