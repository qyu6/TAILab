def pythonx():

    import streamlit as st
    st.markdown('[1]-如何开始codeing前，一键打开经常用的多个网页?')
    code1 = '''
import webbrowser
websites = [
    'www.google.com',
    'www.bing.com',
    'www.yahoo.com'
]
for url in websites:
    webbrowser.open(url)
    '''
    st.code(code1,language='python')


    st.markdown('[2]-如何在服务器中找到某个文件的绝对路径?')
    code2 = '''
import os
current_path = os.path.abspath('<file.name.xxx>')
print(current_path)
        '''
    st.code(code2,language='python')

    st.write('[3]-pip清华镜像源')
    st.code('pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some-package')

    st.write('[4]-pip批量安装依赖第三方库')
    st.code('create "requirement.txt" txt content format:"pandas==0.1.1"\nthen run "pip install -r requirements.txt" in terminal')


def vimx():
    import streamlit as st
    st.markdown('bash终端打开文件')
    st.code('vim xx')

    st.markdown('保存并退出文件')
    st.code(':wq')

    st.markdown('不保存并退出文件')
    st.code(':q!')



def sqlx():
    import streamlit as st
    st.markdown('在bash中启动spark-sql环境')
    st.code('spark-sql')

    st.markdown('显示所有数据库的名字')
    st.code('show databases;')

    st.markdown('选中目标数据库')
    st.code('use <database.name>;')

    st.markdown('显示目标数据库中所有的表')
    st.code('show tables;')

    st.markdown('显示表内所有字段的类型和数值状态')
    st.code('desc <table.name>;')
    
    st.markdown('退出spark-sql shell环境')
    st.code(':quit')

    st.markdown('SQLite3学习笔记:')
    code_sqlite = '''
<sqlite基础命令>
.quit 或 .exit #退出环境
.databases #查看当前数据库
.schema <table.name> #查看表结构
.version #查看版本
.open <database.name>.db #创建数据库(或打开已创建的数据库)
.tables #查看当前数据库所有表
.help #查看命令帮助
.show #查看sqlite命令提示符的默认设置
.mode <xx> #显示表的三种结构list,csv,column
.header on #显示表是否带字段名

如何将本地表导入数据库？
如何将数据库表保存到本地？


<基础表操作：增删改查>
create table <table.name> (id int primary key,name char,age int,sex char); #创建新表(创建新表要先创建/打开数据库，否则表结构不会保存)

insert into <new.table.name> 
select * from <old.table.name>; #将原表中所有的数据复制到一张新表(内容相同，列名不相同); sqlite不支持更换列名，用此方法可以实现更换列名；

create table <new.table.name> 
as select * from <existing.table.name>; #将已存在的表复制另存为一张新表

insert into <table.name> values (0,'zhang0',20,'m'); #基于表的schema插入值

select * from <table.name>; #查找表所有内容

select id,name from <table.name>; #查找id和name字段内容

select * from <table.name> 
where age>25; #where条件查找age>25的信息

select * from <table.name> 
where age>25 and sex='m'; #where条件查找age>25和sex=m的信息

update <table.name> set name='<new.value>' 
where name='<old.value>'; #基于特定条件修改值

delete from <table.name> 
where name='<old.value>'; #基于特定条件删除特定行

alter table <table.name> add address char; #插入一列地址字段，名称为address，类型为char

update <table.name> set address='beijing'; #把全部地址的数据内容更新为beijing

drop table <table.name>; #删除某一张表

create table <new.table.name> 
as select id,name,sex from <old.table.name>; #sqlite不支持删除列，此命令用于选取旧表中几列复制并创建一个新表

alter table <old.table.name> to <new.table.name>; #用于将旧表重命名

select count(*) from <table.name>; #统计表有多少行数据

create, select, insert, update, delete, drop #SQLite最常用的基础命令

select distinct <col.name> from <table.name>; #返回某列的非重复值(去重)，可多列

select *from <table.name> limit <x>; #返回前x行的数据

select *from <table.name> limit <x> offset <y>; #返回前x行的数据,从第y+1行开始返回
注释方式：1. #... 2./*...*/

select * from <table.name> 
order by <col.name>; #按某一列升序排序，默认升序排列 asc(可多列)

select * from <table.name> 
order by <col.name> desc; #按某一列降序排序

select * from <table.name> 
where <col.name>='x'; #搜索列值为x的全部信息(where子句操作符命令：=等于 | <>,!= 不等于 | <小于 | <=小于等于 | !<不小于 | >大于 |>=大于等于 | !> 不大于 | Between x and y 在指定的x和y之间 | is null 为空)

select <col.name> 
where <condition.a> or <condition.b>; #多重条件筛选：And比or的优先级更高；括号比And/or的优先级更高。结论—常用括号来区分优先级

select <col.name> from <table.name> 
where <col.name> in ('value1','value2') 
order by <col.name>; #通过IN括号内的有效值来进行条件筛选

select <col.name> from <table.name> 
where not <col.name>='value' 
order by <col.name> #通过NOT排除某些条件进行筛选


<通配符 %>
select <col.name> from <table.name> 
where <col.name> like 'keywords%'; #通配符LIKE+%,筛选以keywords开头的值(词语和数字均可)

select <col.name> from <table.name> 
where <col.name> like '%keywords%'; #通配符LIKE+%,筛选值内任何位置包含keywords的结果

select <col.name> from <table.name> 
where <col.name> like 'F%y'; #通配符LIKE+%,筛选值以F开头,以y结尾
<通配符 _>

select <col.name> from <table.name> 
where name like 'xxx_'; #通配符LIKE + 下划线_ (_与%用途一样，但一个下划线只能匹配一个字符，多个字符需要对应添加多个下划线)


<拼接字段>
select <col.name1> || '-' || <col.name2> from <table.name>; # 通过连接符将两列的内容通过‘-’连接生成新的字段. ||为连接符，字段值外的连接符用''添加


<度量值计算,计算结果定义别名-new.col.name>
select <col.name1>,<col.name2>,<col.name1>*<col.name2> as <new.col.name> from <table.name>; #通过计算两列值相乘得到新列，基础度量操作+-*/


<文本处理函数>
select upper(<col.name1>) as <new.col.name> from <table.name>; #upper将col1的内容统一全部转化为大写字母。sql常用文本处理函数：upper()-字符串转化为大写，lower()-字符串转化为小写，soundex()-将任何文本串转化为
描述其语音表示的字母数字模式的算法，对字符串发音的相似性进行比较的算法


<日期和时间处理函数>
select order_num from orders 
where strftime('%Y',order_date) = 'xx'; #sqlite专有形式


<汇总计算数据>
select avg(<col.name>) as <new.col.name> from <table.name>; #计算某一字段均值。常用聚集函数：avg(),count(),max(),min(),sum()


<聚集不同汇总值，同时显示>
select count(*) as <new.col1>, max(<col.name2>) as <new.col2>, min(<col.name3>) as <new.col3> from <table.name>; #同时显示多个维度的计算结果值


<分组数据|Group by>
select <col.name>,count(*) as <new.col.name> 
from <table.name> group by <col.name>; #通过group by按照指定维度进行聚合和统计分组


<过滤分组>
select <col.name>,count(*) as <new.col.name> 
from <table.name> group by <col.name> having count(*)>= xx; #通过group by按照指定维度进行聚合和统计分组,并进行统计过滤


<Select子句顺序>
select - from - where - group by - having - order by


<子查询 - 作为where条件>
select <col.name> from <table.name> 
where <col.name1> in (select <col.name2> 
from <table.name1> where <col.name3>='xx') #基于表1的查询结果，作为输入条件应用在表2的查询中，构成子查询(子查询可以是多重，不一定只有1个子查询)


<子查询2 - 作为select条件,示例>
select cust_name,cust_state,
	(select count(*) from orders 
where orders.cust_id = 		customer.cust_id) as orders 
from customers 
order by cust_name; #将子查询作为select的条件加入查询中


<联结表 - join，多表拼接>
为什么要设计多表? 为了更有效的存储，方便地处理；即更好的可伸缩性(scale well)-能够适应不断增加的数据量而不失败
select <col.name1>,<col.name2>,<col.name3> 
from <table.name1>,<table.name2> 
where <table.name1>.<col.name1> = <table.name2>.<col.name2>;
#col.name 1,2,3是来自不同表的join字段，不能来自于同一张表；where条件中可以不是primary key,但要注意字段中的重复情况，重复会进行排列组合，让数据加倍。没有where的条件时，返回的结果为笛卡尔积；

<内联结 - inner join,也叫等值联结equijoin>,示例：
select vend_name,prod_name,prod_price
from vendors
where join products on vendors.vend_id = products.vend_id;

<联结多个表>,示例:
select prod_name,vend_name,prod_price,quantity
from orderitems,products,vendors
where products.vend_id = vendors.vend_id
	and orderitems.prod_id = products.prod_id
	and order_num = 20007;
	
<子查询优化 -> 联结多表>,示例:
select cust_name,cust_contact
from customers
where cust_id in (select cust_id
                 form orders
                 where order_num in (select order_num
                                     from orderitems
                                     where prod_id = 'RGAN01';
                 					)                 
                 );
#优化后↓
select cust_name,cust_contact
from customers,orders,orderitems
where customers.cust_id = orders.cust_id
	and orderitems.order_num = orders.order_num
	and prod_id = 'RGAN01';
	
	
<高级联结 - 表别名(优势:缩短语句，多次引用)>,示例：
select cust_name, cust_contact
from customers as c,orders as o,orderitems as oi
where c.cust_id = o.cust_id
	and oi.order_num = o.order_num
	and prod_id = 'RGAN01';
	
<高级联结 - 自联结self-join>:
select cust_id,cust_name,cust_contact
from customers
where cust_name = (select cust_name
                  from customers
                  where cust_contact = 'Jim Jones');
#优化后↓(性能更佳，自联结通常作为外部语句，用来替代从相同表中检索数据而使用的子查询语句)
select c1.cust_id,c1.cust_name,c1.cust_contact
from customers as c1,customers as c2
where c1.cust_name = c2.cust_name
	and c2.cust_contact = 'Jim Jones';

<高级联结 - 自然联结natural-join>: #(通配符对第一张表使用，所有其他列明确列出，所有没有重复的列被检索出来)
select c.*, o.order_num, o.order_date, oi.prod_id, oi.quantity, oi.item_price
from customers as c, orders as o, orderitems as oi
where c.cust_id = o.cust_id
	and oi.order_num = o.order_num
	and prod_id = 'RGAN01';
	
<高级联结 - 外联结outer-join>: #(统计“被关联+目标表没被关联到”的数据信息.使用外联结时，必须指定包括其所有行的表(排除联结上的行之外的所有行)，right-指outer join右边的表，left-指outer join左边的表. sqlite中不支持right outer，可以通过调整from-where表的先后顺序来实现。)
#(内联结)
select customers.cust_id,orders.order_num
from customers
inner join orders on customers.cust_id = orders.cust_id;
#(外联结)
select customers.cust_id, orders.order_num
from customers
left outer join orders on customers.cust_id = orders.cust_id;

<高级联结 - 带聚集函数的联结>,示例-检索所有顾客及每个顾客所下的订单数
#(方法1)
select customers.cust_id,count(orders.order_num) as num_ord
from customers
inner join orders on customers.cust_id = orders.cust_id
group by customers.cust_id
#(方法2)
select customers.cust id,count(orders.order_num) as num_ord
from customers
left outer join orders on customers.cust_id = orders.cust_id
group by customers.cust_id;


<组合查询 - union> - 多条select语句合并查询/union中各条select每列都为相同字段
select cust_name, cust_contact, cust_email
from customers
where cust_state in ('IL','IN','MI')
union
where cust_name, cust_contact, cust_email
from customers
where cust_name = 'Fun4All'

<union默认将重复的行会被自动取消，如果不需要取消，用union all>
select cust_name, cust_contact, cust_email
from customers
where cust_state in ('IL','IN','MI')
union all
where cust_name, cust_contact, cust_email
from customers
where cust_name = 'Fun4All'

<union中的order by只能作用于最后一条select语句之后>
select cust_name, cust_contact, cust_email
from customers
where cust_state in ('IL','IN','MI')
union
where cust_name, cust_contact, cust_email
from customers
where cust_name = 'Fun4All'
order by cust_name, cust_contact


<插入数据 - insert> - 插入完整的行,每个值将按默认顺序插入
insert into customers
value('123','abc','thooo');

<insert - 更保险的方式,分别给定列名和值，一一对应>
insert into customers(cust_id,cust_contact,cust_name)
value('123','abc','thooo')

<insert 检索后的结果数据 - customers.new与customer表结构相同，但主键cust_id不能相同>
insert into customers(cust_id,cust_contact,cust_name)
select cust_id,cust_contact,cust_name
from customers.new

<从一个表复制到另一个表>
create table table.new as select * from old.table;


<更新和删除数据 - insert/delete> - 注意跟where条件，否则更新全表
update customers
set cust_email = 'newemail@xxx.com'
where cust_id = '1000000';

<更新多列>
update customers
set cust_email = 'newemail@xxx.com',cust_contact='bob'
where cust_id = '1000000';

<删除某列的值> - 删除cust_email列
update customers
set cust_email=null
where cust_id='1000000';

<删除行>
delect customers
where cust_id='100000';

<创建表> - 表名，表定义
create table product
(
	prod_id	char(10) not null,
    vend_id char(10) not null,
    prod_name decimal(8,2) not null,
)

<更新表 - alert> - 增加一列
alter table vendors
add vend_phone char(20);

<alert - 删除一列>
alter table vendors
drop column vend_phone;

<删除表>
drop table custcopy;


<视图-view|虚拟表> 作用：重用sql语句，简化复杂sql操作，使用表的一部分而不是整体，权限控制和分享等
#利用视图简化复杂的联结
create view productcustomers as 
select cust_name,cust_contact,prod_id
from customers,orders,orderitems
where customers.cust_id=orders.cust_id
	and orderitems.order_num=orders.order_num;
	
select cust_name,cust_contact
from productcustomers
where prod_id ='RGAN01';

<使用视图创建常用的数据格式>
create view vendorlocation as
select rtrim(vend_name) + '(' + RTRIM(vend_country)+')'
	as vend_title
from vendors;

<使用视图过滤不想要的数据>
create view customeremailist as
select cust_id, cust_name, cust_email
from customers
where cust_email is not null;

<使用视图检索计算结果>
create view orderitemexpanded as
select order_num, prod_id, quantity, item_price, quantity*item_price as expanded_price
from orderitems

select * 
from orderitemexpanded
where order_num=20008;
    '''
    st.code(code_sqlite,language='sql')


def gitx():
    import streamlit as st
    st.markdown('返回上级目录')
    st.code('cd ..')

    st.markdown('进入到某级目录')
    st.code('code xx')
    
    st.markdown('显示当前路径')
    st.code('pwd')
    
    st.markdown('清屏')
    st.code('clear')
    
    st.markdown('列出当前路径下文件')
    st.code('ls')
    
    st.markdown('列出文件的详细信息')
    st.code('ls -l')

    st.markdown('列出路径下所有的文件，包含隐藏文件')
    st.code('ls -a')

    st.markdown('创建一个文件夹目录')
    st.code('mkdir xx')

    st.markdown('创建一个新文件')
    st.code('touch xx.xx')

    st.markdown('移除一个文件')
    st.code('rm xx.xx')

    st.markdown('移除一个文件夹')
    st.code('rm -r xx')

    st.markdown('强制移除某个文件或文件夹')
    st.code('rm -rf xx (never use "rm -rf", will delete everthing.)')

    st.markdown('* x1文件,x2文件夹：移动文件x1到文件夹x2\n'
    '* x1文件夹,x2不存在,重命名文件夹x1为x2\n'
    '* x1文件夹,x2文件夹,将文件夹x1移动到文件夹x2内')
    st.code('mv x1 x2')

    st.markdown('显示所有命令记录历史')
    st.code('history')

    st.markdown('显示帮助文档')
    st.code('help xx')

    st.markdown('退出bash环境')
    st.code('exit')

    st.markdown('显示当前git配置信息')
    st.code('git config -l')
    
    st.markdown('显示当前系统的git配置信息')
    st.code('git config --system --list')

    st.markdown('显示全局配置的内容，用户名+账户等，必须配置')
    st.code('git config --global --list')

    st.markdown('* git系统配置文件\n'
    '* git全局配置文件路径')
    st.code('c:\program\files\git\etc\gitconfig\nc:\.gitconfig')
    
    st.markdown('设置用户名和邮箱-用户标识')
    st.code('git config --global --user.name "xx"\ngit config --global --user.email "xx"')

    st.markdown('显示git常用的命令')
    st.code('git')

    st.markdown('查看之前的版本历史信息')
    st.code('git log')
    
    st.markdown('简化git版本历史输出的信息')
    st.code('git log --pretty=online')

    st.markdown('在当前路径初始化一个git项目')
    st.code('git init')

    st.markdown('显示当前路径下所有文件的git状态')
    st.code('git status')

    st.markdown('将路径下所有文件添加到stage区')
    st.code('git add .')

    st.markdown('提交stage区的内容到本地仓库(如报错检查gitconfig信息)')
    st.code('git commit -m "<commit.message>"')

    st.markdown('将本地仓库内容同步到远程仓库')
    st.code('git push')

    st.markdown('将远程仓库克隆到本地')
    st.code('git clone <url>')

    st.markdown('将最新的远程仓库同步到本地')
    st.code('git pull')

    st.markdown('git忽略本地项目更改，直接用服务器版本来覆盖本地仓库')
    st.code('git fetch --all\ngit reset --hard origin/master\ngit pull')

    st.markdown('显示当前项目所有分支')
    st.code('git branch')

    st.markdown('显示所有远程分支')
    st.code('git branch -r')

    st.markdown('新建一个分支并停留在当前分支')
    st.code('git branch <branch.name>')

    st.markdown('新建一个分支并切换到该分支上')
    st.code('git checkout -b <branch.name>')

    st.markdown('转换工作区分支到新创建的分支上')
    st.code('git switch <branch.name>')

    st.markdown('如果remote仓库没有这个branch，则这个分支直接git push会无效，可以用这个命令来实现分支push')
    st.code('git push --set-upstream origin <branch.name>')

    st.markdown('合并指定分支到当前分支上')
    st.code('git merge <branch.name>')

    st.markdown('删除远程分支')
    st.code('git push origin --delete <branch.name>\ngit branch -dr <remote/branch>')

    st.markdown('显示当前路径下所有远程库')
    st.code('git remove -v')
    
    st.markdown('查看本机是否安装ssh')
    st.code('ssh')

    st.markdown('生成ssh公钥')
    st.code('ssh-keygen')

    st.markdown('使用加密算法生成公钥，一路回车;c:/admin/.ssh下回生成两个文件, .pub后缀的文件打开后，与Gitlab/Github绑定')
    st.code('ssh-keygen -t ras')
    
    st.markdown('git端验证绑定ssh是否成功')
    st.code('ssh -T git@github.com | ssh -T git@gitlab.com ..')

    st.markdown('输出所有git提交历史日志，查找要退回的版本')
    st.code('git log')

    st.markdown('进行版本回退操作')
    st.code('git reset --hard <commit_id>')
    
    st.markdown('回滚后推送至远程分支')
    st.code('git push origin')

    st.markdown('快捷命令，回退到上个版本')
    st.code('git reset --hard HEAD^ | HEAD^^-previous~previous')

    st.markdown('进入python交互模式')
    st.code('python -i')
    
    st.markdown('进入python终端命令行交互模式')
    st.code('winpty python')

    st.markdown('退出终端python交互模式')
    st.code('quit()')

    st.markdown('bash中运行.py文件，运行完成后停留在python终端环境')
    st.code('python -i xx.py')
    
    st.markdown('bash中运行.py文件，运行完成后退出python终端环境')
    st.code('winpty python xx.py')

    # st.markdown('')
    # st.code('')
    
    # st.markdown('')
    # st.code('')
