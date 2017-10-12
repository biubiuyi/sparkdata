sparkdata
=
整理了实习半年以来的个人工作代码以及学习到的知识点

---
 * crawler: 解析爬虫组爬到的数据到hdfs中
 * FeatureRobot: 特征机器人相关代码,包括特征交叉,筛选,GBDT模型造新特征
 * fraud_item: 电商,社交异常数据筛选,包括使用规则和模型筛选
 * taobao_ciyun: 电商用户购买商品词云数据更新,包括分词,词语清洗,tfidf权重计算
 * weibo_analysis: 微博用户关系网路构建，计算用户pagerank值。用到了graphlab工具包（学生申请使用免费，目前已到期）
 
---
  markdown文档撰写说明:https://github.com/guodongxiaren/README#readme
  
  git使用说明:http://rogerdudler.github.io/git-guide/index.zh.html
  git版本回退：http://www.cnblogs.com/cposture/p/git.html
---
更新本地文件到github已有库（本地已有git库）：
 如果还没有克隆现有仓库，并欲将现有本地git仓库连接到某个远程服务器，可以使用如下命令添加：
 git remote add origin <server>
 如此就能够将本地库改动推送到所添加的服务器上去


如果以及连接好远端库，执行更新与合并操作：
1. git pull ；将远端的代码拉到本地，先同步远端库代码
2. git add * ；把修改文件添加到暂存区index，临时保存改动。
3. git commit -m '说明' ；把修改代码提交到本地HEAD（指向最后一次提交结果）
4. git push origin master ;将修改代码提交到远端库
  
  
  
