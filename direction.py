import pandapower as pp
import pandapower.networks as pn


net = pn.case118()
print(net.line[['from_bus', 'to_bus']])  # 一条输电线的入端和出端

pp.runpp(net)
print(net.res_line[['p_from_mw', 'p_to_mw']])  # 一条输电线在入端和出端的功率，正为流入，负为流出，线路上会有热耗，所以输入功率>输出功率
# 比如上述例子的0号输电线，虽然打印的第一个列表中，入端定义为0号母线，出端为1号母线，初始方向为0->1
# 但经过潮流计算后，在第二个列表中0号母线的功率为负，1号母线功率为正，因此实际潮流方向是从1->0


net.res_trafo[['p_hv_mw', 'p_hv_mw']]