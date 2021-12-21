# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 02:25:09 2021

@author: Yunxiang
"""

import simpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd
import time

def generate_interval(A,n,rho,mu):
    if A == 'M':
        return random.expovariate(n*rho*mu)

def generate_service(B,mu):
    if B == 'M':
        return random.expovariate(mu)
    if B == 'D':
        return mu
    if B == 'H':
        if np.random.random() < 0.75:
            return random.expovariate(1)
        else:
            return random.expovariate(1/5)
        
    
def store_run(env, servers, A, B, n, rho, mu):
    i = 0
    while True:
        i += 1
        yield env.timeout(generate_interval(A,n,rho,mu))
        env.process(customer(env, i, servers, B, mu))
        # env.process(customer_priority(env, i, servers, B, mu))
        

def customer(env, customer, servers, B, mu):
    service_time = generate_service(B, mu)
    with servers.request() as request:
        t_arrival = env.now
        # print(env.now, f'customer {customer} arrives')
        yield request
        # print(env.now, f'customer {customer} is being served')
        t_wait = env.now
        yield env.timeout(service_time)
        # print(env.now, f'customer {customer} departs')
        # t_sojurn = env.now
        wait_t.append(t_wait - t_arrival)
        
        
        
def main(A,B,n,rho,mu):
    
    env = simpy.Environment()
    servers = simpy.Resource(env, capacity = n)
    env.process(store_run(env, servers, A, B, n, rho, mu))
    # env.process(observe(env, servers))
    env.run(until = 60000)


# # M/M/1
rho_list = [0.7,0.8,0.9,0.95]
n_list = [4]

data = pd.DataFrame(columns = ['run','rho','number of servers','number of customers', 'mean waiting time'])
for n_servers in n_list:
      
    for rho in rho_list:
        
        t0 = time.time()
        for i in range(500):
            wait_t = []
            main('M','M', n=n_servers, rho=rho, mu=1)
            mean_list = np.concatenate((np.arange(100,1000,100), np.arange(1000,10000,1000), np.arange(10000,100001,10000)))
            df = pd.DataFrame({'run': [i]*len(mean_list),
                                'rho': [rho]*len(mean_list),
                                'number of servers': [n_servers]*len(mean_list),
                                'number of customers': mean_list,
                                'mean waiting time': [np.mean(wait_t[:n_customers]) for n_customers in mean_list]})
            data = data.append(df, ignore_index = True)
            # print(len(wait_t))
            
        print(f'{rho} {n_servers} finished')
        print(time.time()-t0)

data.to_csv('rho4.csv')

        
    
# M/M/1 shortest job priority
def customer_priority(env, customer, servers, B, mu):
    service_time = generate_service(B, mu)
    with servers.request(priority = service_time) as request:
        t_arrival = env.now
        # print(env.now, f'customer {customer} arrives')
        yield request
        # print(env.now, f'customer {customer} is being served')
        t_wait = env.now
        yield env.timeout(service_time)
        # print(env.now, f'customer {customer} departs')
        # t_sojurn = env.now
        wait_t.append(t_wait - t_arrival)
        
        
def main_priority(A,B,n,rho,mu):
    
    env = simpy.Environment()
    servers = simpy.PriorityResource(env, capacity = n)
    env.process(store_run(env, servers, A, B, n, rho, mu))
    # env.process(observe(env, servers))
    env.run(until = 30000)



data = pd.DataFrame(columns = ['run','rho','number of servers','number of customers', 'mean waiting time'])
for i in range(500):
    wait_t = []
    main_priority('M','M', n=4, rho=0.9, mu=1)
    mean_list = np.concatenate((np.arange(100,1000,100), np.arange(1000,10000,1000), np.arange(10000,100001,10000)))
    df = pd.DataFrame({'run': [i]*len(mean_list),
                        'rho': [0.9]*len(mean_list),
                        'number of servers': [2]*len(mean_list),
                        'number of customers': mean_list,
                        'mean waiting time': [np.mean(wait_t[:n_customers]) for n_customers in mean_list]})
    data = data.append(df, ignore_index = True)
    print(len(wait_t), i)
data.to_csv('sjp4.csv')


# M/D/1 M/D/n

data = pd.DataFrame(columns = ['run','rho','number of servers','number of customers', 'mean waiting time'])
for i in range(500):
    wait_t = []
    main('M','D', n=4, rho=0.9, mu=1)
    mean_list = np.concatenate((np.arange(100,1000,100), np.arange(1000,10000,1000), np.arange(10000,100001,10000)))
    df = pd.DataFrame({'run': [i]*len(mean_list),
                        'rho': [0.9]*len(mean_list),
                        'number of servers': [4]*len(mean_list),
                        'number of customers': mean_list,
                        'mean waiting time': [np.mean(wait_t[:n_customers]) for n_customers in mean_list]})
    data = data.append(df, ignore_index = True)
    print(len(wait_t), i/5)
data.to_csv('md4.csv')



# M/H/1 
data = pd.DataFrame(columns = ['run','rho','number of servers','number of customers', 'mean waiting time'])
for i in range(500):
    wait_t = []
    main('M','H', n=4, rho=0.9, mu=0.5)
    mean_list = np.concatenate((np.arange(100,1000,100), np.arange(1000,10000,1000), np.arange(10000,100001,10000)))
    df = pd.DataFrame({'run': [i]*len(mean_list),
                        'rho': [0.9]*len(mean_list),
                        'number of servers': [4]*len(mean_list),
                        'number of customers': mean_list,
                        'mean waiting time': [np.mean(wait_t[:n_customers]) for n_customers in mean_list]})
    data = data.append(df, ignore_index = True)
    print(len(wait_t), i)
data.to_csv('mh4.csv')

data = pd.DataFrame(columns = ['run','rho','number of servers','number of customers', 'mean waiting time'])
for i in range(500):
    wait_t = []
    main('M','M', n=1, rho=0.9, mu=0.5)
    mean_list = np.concatenate((np.arange(100,1000,100), np.arange(1000,10000,1000), np.arange(10000,100001,10000)))
    df = pd.DataFrame({'run': [i]*len(mean_list),
                        'rho': [0.9]*len(mean_list),
                        'number of servers': [1]*len(mean_list),
                        'number of customers': mean_list,
                        'mean waiting time': [np.mean(wait_t[:n_customers]) for n_customers in mean_list]})
    data = data.append(df, ignore_index = True)
    print(len(wait_t), i/5)
data.to_csv('mm1_mu.csv')






