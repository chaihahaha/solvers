import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

def read_input_file(filename):
    """读取输入文件"""
    try:
        df = pd.read_csv(filename, sep='\t')
        # 去除可能的前后空格
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def calculate_shipping_cost(weight, num_items, total_price):
    """计算单个订单的运费"""
    base = weight + num_items * 0.3 - 8
    if total_price > 299:
        return base
    else:
        return base + 15

def solve_optimal_partition(df, max_orders=None):
    """
    求解最优分区问题
    
    Args:
        df: 包含商品信息的DataFrame
        max_orders: 最大订单数，如果为None则自动估计
    """
    
    # 商品数量
    n_items = len(df)
    
    # 如果没有指定最大订单数，则使用启发式估计
    if max_orders is None:
        # 最少订单数：至少需要 ceil(总件数 / 平均每单能放的数量)
        total_quantity = df['个数'].sum()
        # 简单估计：假设平均每单可放4-6件商品
        max_orders = min(total_quantity, max(2, int(total_quantity / 4) + 1))
    
    # 创建模型
    model = gp.Model("OptimalMerchandisePartition")
    
    # 决策变量
    # x[i,k]: 商品i分配到订单k的数量
    x = {}
    for i in range(n_items):
        for k in range(max_orders):
            x[i, k] = model.addVar(
                vtype=GRB.INTEGER, 
                lb=0, 
                ub=df.iloc[i]['个数'],
                name=f"x_{i}_{k}"
            )
    
    # 辅助变量
    # weight[k]: 订单k的总重量
    weight = {}
    # num_items[k]: 订单k的商品数量
    num_items = {}
    # total_price[k]: 订单k的总价格
    total_price = {}
    # shipping_cost[k]: 订单k的运费
    shipping_cost = {}
    # order_used[k]: 订单k是否被使用（0/1）
    order_used = {}
    # above_threshold[k]: 订单k总价是否超过299（0/1）
    above_threshold = {}
    
    for k in range(max_orders):
        weight[k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"weight_{k}")
        num_items[k] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"num_items_{k}")
        total_price[k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"total_price_{k}")
        shipping_cost[k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"shipping_cost_{k}")
        order_used[k] = model.addVar(vtype=GRB.BINARY, name=f"order_used_{k}")
        above_threshold[k] = model.addVar(vtype=GRB.BINARY, name=f"above_threshold_{k}")
    
    # 目标函数：最小化总运费
    model.setObjective(gp.quicksum(shipping_cost[k] for k in range(max_orders)), GRB.MINIMIZE)
    
    # 约束条件
    
    # 1. 所有商品必须全部分配
    for i in range(n_items):
        model.addConstr(
            gp.quicksum(x[i, k] for k in range(max_orders)) == df.iloc[i]['个数'],
            f"assign_all_{i}"
        )
    
    # 2. 计算每个订单的指标
    for k in range(max_orders):
        # 总重量
        model.addConstr(
            weight[k] == gp.quicksum(x[i, k] * df.iloc[i]['重量'] for i in range(n_items)),
            f"weight_constraint_{k}"
        )
        
        # 商品数量
        model.addConstr(
            num_items[k] == gp.quicksum(x[i, k] for i in range(n_items)),
            f"num_items_constraint_{k}"
        )
        
        # 总价格
        model.addConstr(
            total_price[k] == gp.quicksum(x[i, k] * df.iloc[i]['单价'] for i in range(n_items)),
            f"total_price_constraint_{k}"
        )
    
    # 3. 定义order_used变量（如果有任何商品，则订单被使用）
    for k in range(max_orders):
        # 如果num_items[k] > 0，则order_used[k] = 1
        M = 1000  # 大M常数
        model.addConstr(num_items[k] <= M * order_used[k], f"order_used_lower_{k}")
        model.addConstr(num_items[k] >= 0.1 * order_used[k], f"order_used_upper_{k}")
    
    # 4. 定义above_threshold变量
    for k in range(max_orders):
        # 如果total_price[k] > 299，则above_threshold[k] = 1
        M_price = 10000  # 价格的大M常数
        # 当above_threshold[k] = 0时，total_price[k] <= 299
        # 当above_threshold[k] = 1时，total_price[k] >= 299.001
        model.addConstr(total_price[k] <= 299 + M_price * above_threshold[k], f"price_upper_{k}")
        model.addConstr(total_price[k] >= 299.001 - M_price * (1 - above_threshold[k]), f"price_lower_{k}")
    
    # 5. 运费计算
    for k in range(max_orders):
        # 运费 = (总重量 + 商品数量*0.3 - 8) + 15 * (1 - above_threshold[k])
        # 即：如果总价>299，above_threshold[k]=1，则运费 = base
        #     否则，运费 = base + 15
        base_cost = weight[k] + 0.3 * num_items[k] - 8
        penalty = 15 * (1 - above_threshold[k])
        model.addConstr(shipping_cost[k] == base_cost + penalty, f"shipping_cost_{k}")
    
    # 6. 如果没有商品，则above_threshold必须为0
    for k in range(max_orders):
        model.addConstr(above_threshold[k] <= order_used[k], f"no_empty_above_threshold_{k}")
    
    # 设置求解参数
    model.Params.TimeLimit = 300  # 5分钟时间限制
    model.Params.MIPGap = 0.01  # 1%的优化间隙
    model.Params.Threads = 8  # 使用8个线程
    
    # 求解
    model.optimize()
    
    # 检查求解状态
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        print(f"求解完成，目标值: {model.objVal:.2f}")
        if model.status == GRB.TIME_LIMIT:
            print("已达到时间限制，返回当前最优解")
    else:
        print(f"求解失败，状态: {model.status}")
        return None
    
    # 提取结果
    results = []
    for k in range(max_orders):
        order_items = []
        total_items_in_order = 0
        
        for i in range(n_items):
            quantity = int(x[i, k].x + 0.5)  # 四舍五入
            if quantity > 0:
                item_data = {
                    '商品名': df.iloc[i]['商品名'],
                    '单价': df.iloc[i]['单价'],
                    '重量': df.iloc[i]['重量'],
                    '个数': quantity
                }
                order_items.append(item_data)
                total_items_in_order += quantity
        
        if order_items:
            # 计算订单的统计信息
            order_weight = sum(item['重量'] * item['个数'] for item in order_items)
            order_price = sum(item['单价'] * item['个数'] for item in order_items)
            actual_shipping = calculate_shipping_cost(order_weight, total_items_in_order, order_price)
            
            order_summary = {
                '订单编号': k + 1,
                '商品列表': order_items,
                '总重量': order_weight,
                '总价格': order_price,
                '商品数量': total_items_in_order,
                '运费': actual_shipping
            }
            results.append(order_summary)
    
    return results

def output_partitions(results, input_filename, output_filename=None):
    """输出分区结果"""
    if output_filename is None:
        output_filename = 'output_partitions.txt'
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        for i, order in enumerate(results):
            f.write(f"订单 {order['订单编号']}: 总价={order['总价格']:.2f}, 运费={order['运费']:.2f}\n")
            f.write("商品名\t单价\t重量\t个数\n")
            
            for item in order['商品列表']:
                f.write(f"{item['商品名']}\t{item['单价']}\t{item['重量']}\t{item['个数']}\n")
            
            f.write("\n")
    
    # 同时输出到控制台
    print("\n最优分区结果：")
    print("=" * 60)
    for i, order in enumerate(results):
        print(f"订单 {order['订单编号']}:")
        print(f"  总价: {order['总价格']:.2f}, 重量: {order['总重量']:.2f}, 商品数: {order['商品数量']}, 运费: {order['运费']:.2f}")
        
        df_order = pd.DataFrame(order['商品列表'])
        print(df_order.to_string(index=False))
        print()
    
    total_shipping = sum(order['运费'] for order in results)
    print(f"总运费: {total_shipping:.2f}")
    print(f"详细结果已保存到: {output_filename}")
    
    return output_filename

def main():
    # 读取输入文件
    filename = input("请输入输入文件名（默认为input.txt）: ") or "input.txt"
    df = read_input_file(filename)
    
    if df is None:
        print("无法读取文件，请检查文件格式和内容。")
        print("文件应为制表符分隔，包含列：商品名、单价、重量、个数")
        return
    
    print("读取到的商品信息：")
    print(df)
    print()
    
    # 设置最大订单数
    total_items = df['个数'].sum()
    max_orders = int(input(f"请输入最大订单数（商品总数为{total_items}，建议值{min(total_items, 10)}）: ") or min(total_items, 10))
    
    # 求解
    print("正在求解，请稍候...")
    results = solve_optimal_partition(df, max_orders)
    
    if results:
        output_filename = input("请输入输出文件名（默认为output.txt）: ") or "output.txt"
        output_partitions(results, filename, output_filename)
    else:
        print("无解，请尝试增加最大订单数或检查输入数据。")

if __name__ == "__main__":
    main()
