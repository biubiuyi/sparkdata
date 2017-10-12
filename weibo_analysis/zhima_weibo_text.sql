#=========1:weibo usr info feature


#zhima_score_3w table:
wlcredit.t_zlj_zhima20170110_userinfo_snwb

#weibo user info table:
wlbase_dev.t_base_weibo_user_new

#3w zhima_usr weibo infor table
wlservice.t_lt_zhima_weibo_usrinfo_3w

#3w zhima_usr weibo text table
wlservice.t_wrt_3wuser_weibotext;

# all sina users occupation infor
wl_base.t_base_weibo_employment_info

#map join the weibo usrinfo with zhima_usr
create table wlservice.t_lt_zhima_weibo_usrinfo_3w as
	SELECT  /*+ mapjoin(t2)*/
	t2.zhima_score,t2.snwb,t1.name,t1.province,
	t1.city,t1.location,t1.description,t1.gender,
	t1.followers_count,t1.friends_count,t1.statuses_count,
	t1.favourites_count,t1.created_at,t1.allow_all_act_msg,t1.geo_enabled,
	t1.verified,t1.verified_type,t1.allow_all_comment,
	t1.online_status,t1.bi_followers_count
	from
	wlcredit.t_zlj_zhima20170110_userinfo_snwb t2
	left join wlbase_dev.t_base_weibo_user_new t1 
	on t1.id=t2.snwb
	where ds=20161213

#========weibo text statistic
#1) filter numeric columns from the weibo_text table
create table wlservice.t_lt_weibo_text_num_columns as
select mid,user_id,reposts_count,comments_count,attitudes_count,favorited,thumbnail_pic,islongtext,weibo_type,
case when 
retweeted_status = '-' then '1'
else '0' end as isOriginal
from wlservice.t_wrt_3wuser_weibotext


create table wlservice.t_lt_weibo_text_desrcibe as
select user_id, 
sum(reposts_count) as sum_reposts,
sum(comments_count ) as sum_comments,
sum(attitudes_count) as sum_attitudes,
sum(favorited) as sum_favorited,
sum(case when thumbnail_pic='-' then 0 else 1 end) as sum_picture,
sum(islongtext) as sum_longtext,
sum(isoriginal) as sum_original,
avg(reposts_count) as avg_reposts,
avg(comments_count ) as avg_comments,
avg(attitudes_count) as avg_attitudes,
avg(favorited) as avg_favorited,
avg(islongtext) as avg_longtext,
avg(isoriginal) as avg_orig,
sum(weibo_type) as weibo_special,
count(1) as count_weibo
from wlservice.t_lt_weibo_text_num_columns
group by user_id;

#2)filter text columns 
create table wlservice.t_lt_weibo_zhengwen_columns as
select mid,user_id,created_at,text,source,weibo_type,retweeted_status
from wlservice.t_wrt_3wuser_weibotext

#=======load local data to hive
#/opt/cloudera/parcels/CDH/lib/hadoop/bin/hadoop fs -put dataFilePath hdfs://10.3.4.220:9600/user/lt/
create table wlservice.t_lt_zhima_score_20w 
	(weboid string,score int);

load data inpath '/user/lt/zhima_score_hive' 
overwrite into table wlservice.t_lt_zhima_score_20w



#======download file from hive
#/opt/cloudera/parcels/CDH/lib/hadoop/bin/hadoop fs -cat hdfs://10.3.4.220:9600/hive/warehouse/wlservice.db/t_lt_weibo_text_desrcibe/* >weibo_text_3w.csv 

#join
create table wlservice.t_lt_zhima_score_weibo_text_desrcibe as
	select * from 
	wlservice.t_lt_zhima_score_20w A
	left join
	wlservice.t_lt_weibo_text_desrcibe B
	on A.weboid = B.user_id


CREATE TABLE  if not exists wl_base.t_base_weibo_employment_info(
uuid String COMMENT '微博id',
employment_info String COMMENT '职业信息',
)
COMMENT '微博用户职业信息'
PARTITIONED BY  (ds STRING )
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\001'  LINES TERMINATED BY '\n' stored as textfile;


