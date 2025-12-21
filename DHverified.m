clear; clc; close all;

%% 1. 定义 DH 参数 (Modified DH)
% 原始数据列顺序: [a(i-1), alpha(i-1), d(i), theta(i)]
% 单位: 长度(mm), 角度(deg)
dh_params_deg = [
    0,      0,    120,    0;
    0,     90,    100,   90;
    0,    -90,      0,   0;
    400,    0,    150,    0;
   -400,  180,   -150,    0;
    0,    -90,   -100,  -90;
    0,    -90,   -120,   0;
];

% 角度转弧度
dh = dh_params_deg;
dh(:,[2, 4]) = deg2rad(dh(:,[2, 4]));

%% 2. 创建 Link 对象列表
% 我们利用 MATLAB 的自动扩展特性，或者使用 Cell 数组来存储。
% 这里最稳妥的方法是使用 Cell 数组暂存，最后转换。

links_cell = cell(1, 7); % 创建一个空的元胞数组

for i = 1:size(dh, 1)
    % 提取参数
    a_val       = dh(i, 1); % a(i-1)
    alpha_val   = dh(i, 2); % alpha(i-1)
    d_val       = dh(i, 3); % d(i)
    offset_val  = dh(i, 4); % theta(i) 的初始偏移
    
    % 创建 Link 对象 (MDH)
    % 注意：'theta' 是关节变量 q，这里的 dh(i,4) 对应的是 'offset'
    L = Link('d', d_val, ...
             'a', a_val, ...
             'alpha', alpha_val, ...
             'offset', offset_val, ...
             'modified'); 
    
    % 设置关节范围
    L.qlim = [-2*pi, 2*pi];
    
    % 【修改点2】：存入 Cell 数组，避免直接对象拼接带来的类型错误
    links_cell{i} = L; 
end

% [links_cell{:}] 会将元胞里的内容取出并排列成一个 Link 向量
links = [links_cell{:}];

%% 3. 创建 SerialLink 机器人对象
robot = SerialLink(links, 'name', '7-DOF Robot (MDH)');

% 显示 DH 参数表详情
disp('RTB 生成的机器人 DH 参数表:');
robot.display();

%% 4. 可视化
figure('Name', 'RTB 7-DOF Visualization', 'Color', 'w');

% 定义零位姿态
q_zero = zeros(1, 7);

% 绘制机器人
w_range = [-1000 1000 -1000 1000 -500 1500];

robot.plot(q_zero, 'workspace', w_range, 'view', [45 30], 'scale', 0.5);

% 简单的示教
robot.teach(q_zero, 'workspace', w_range);