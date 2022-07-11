import json
import time
import numpy as np
import os

class GraphResultCollector():
    def __init__(self, graph_name):
        self.best_mis = None
        self.best_mis_time = None
        self.best_mis_size = 0
        self.total_solutions = 0
        self.results = {}
        self.graph_name = graph_name.replace(".mat", "") # do not use splitext here, due to recursive calls!

    def start_timer(self):
        self.start_time = time.monotonic()

    def start_process_timer(self):
        self.process_start_time = time.process_time()

    def collect_result(self, mis):
        mis_len = np.ravel(mis).shape[0] 
        if mis_len > self.best_mis_size:
            self.best_mis_time = time.monotonic() - self.start_time
            self.best_mis_process_time = time.process_time() - self.process_start_time
            self.best_mis = mis
            self.best_mis_size = mis_len

    def add_iteration(self):
        self.total_solutions += 1

    def stop_timer(self):
        self.total_time = time.monotonic() - self.start_time

    def finalize(self):
        if self.best_mis is not None:
            return {
                "found_mis": True,
                "vertices": self.best_mis_size,
                "solution_time": self.best_mis_time,
                "mis": np.ravel(self.best_mis).tolist(),
                "total_solutions": self.total_solutions,
                "total_time": self.total_time,
                "solution_process_time": self.best_mis_process_time
            }
        else:
            return {
                "found_mis": False,
                "total_solutions": self.total_solutions,
                "total_time": self.total_time
            }

    def __add__(self, gcol):
        if self.graph_name != gcol.graph_name:
            raise Exception("Trying to merge two graph collectors of different graphs")

        res = GraphResultCollector(self.graph_name)
        res.total_solutions = self.total_solutions + gcol.total_solutions
        res.total_time = max(self.total_time, gcol.total_time)
           
        if self.best_mis is None and gcol.best_mis is not None:
            self.best_mis_size = -1
       
        if gcol.best_mis is None and self.best_mis is not None:
            gcol.best_mis_size = -1

        if gcol.best_mis is None and self.best_mis is None:
            return res

        if self.best_mis_size > gcol.best_mis_size:
            res.best_mis = self.best_mis
            res.best_mis_size = self.best_mis_size
            res.best_mis_time = self.best_mis_time
            res.best_mis_process_time = self.best_mis_process_time
        else:
            res.best_mis = gcol.best_mis
            res.best_mis_size = gcol.best_mis_size
            res.best_mis_time = gcol.best_mis_time
            res.best_mis_process_time = gcol.best_mis_process_time

        return res
            

class ResultCollector():
    def __init__(self):
        self.collectors = []
        self.current_collector = None

    def new_collector(self, graph_name):
        if self.collectors:
            self.current_collector.stop_timer()

        g = GraphResultCollector(graph_name)
        self.collectors += [g]
        self.current_collector = g

        return g

    def finalize(self, out_path):
        if self.current_collector:
            self.current_collector.stop_timer()
        results = {}
        for g in self.collectors:
            results[g.graph_name] = g.finalize()
        self.dump(results, out_path)

    def dump(self, results, out_path):
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, sort_keys = True, indent=4)


collector = ResultCollector()

