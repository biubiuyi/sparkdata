# extract weibo usr info of sample
create table wlservice.t_lt_zhima_weibo_usrinfo_4w as
	SELECT  /*+ mapjoin(t2)*/
	t2.snwb,t2.zhima_score,t1.name,t1.province,
	t1.city,t1.location,t1.description,t1.gender,
	t1.followers_count,t1.friends_count,t1.statuses_count,
	t1.favourites_count,t1.created_at,t1.allow_all_act_msg,t1.geo_enabled,
	t1.verified,t1.verified_type,t1.allow_all_comment,
	t1.online_status,t1.bi_followers_count
	from
	wlcredit.t_zlj_zhima_model_train_v2 t2
	left join wlbase_dev.t_base_weibo_user_new t1 
	on t1.id=t2.snwb
	where ds=20161213

#extract weibo zhengwen of sample
drop table wlservice.t_wrt_3wuser_weibotext;
create table wlservice.t_lt_zhima_weibo_zhengwen_4w as
	SELECT  /*+ mapjoin(t2)*/
	t1.snwb,t1.zhima_score,t2.* 
	from
    wlcredit.t_zlj_zhima_model_train_v2 t1
    join
    (select * from wlbase_dev.t_base_weibo_text where ds = 20161126)t2
    on
    t1.snwb = t2.user_id

#statistic numerical columns
create table wlservice.t_lt_weibo_zhengwen_desrcibe as
select snwb, zhima_score,
sum(reposts_count) as sum_reposts,
sum(comments_count ) as sum_comments,
sum(attitudes_count) as sum_attitudes,
sum(favorited) as sum_favorited,
sum(case when thumbnail_pic='-' then 0 else 1 end) as sum_picture,
sum(islongtext) as sum_longtext,
sum(case when retweeted_status = '-' then '1' else '0' end) as sum_original,
avg(reposts_count) as avg_reposts,
avg(comments_count ) as avg_comments,
avg(attitudes_count) as avg_attitudes,
avg(favorited) as avg_favorited,
avg(islongtext) as avg_longtext,
sum(weibo_type) as weibo_special,
count(1) as count_weibo
from wlservice.t_lt_zhima_weibo_zhengwen_4w
group by user_id;



#extract follows of 4w sample
create table wlservice.t_lt_weibo_follows_4w as
	select /*+ mapjoin(t1)*/
	t1.snwb,t2.ids 
	from
	(select snwb from wlcredit.t_zlj_zhima_model_train_v2)t1
		join wlservice.t_lt_weibo_usr_follows_all t2
		on t1.snwb=t2.id
		

#download data 
#/opt/cloudera/parcels/CDH/lib/hadoop/bin/hadoop fs -cat hdfs://10.3.4.220:9600/hive/warehouse/wlservice.db/t_lt_weibo_text_desrcibe/* >weibo_text_3w.csv 


#join phone tag,paltform,black_phone_field of 4w sample
create table wlservice.t_zlj_zhima_model_train_4w as
	select A.*,B.label as phone_tag,C.platform as daichang_platform,
	D.phone_field as balck_phone
	from
	wlcredit.t_zlj_zhima_model_train_v2 A
	left join 
	wlcredit.t_credit_phone_tag_sogou_label B
	on A.tel =B.phone
	left join
	(select * from wl_base.t_base_yixin_daichang
		where ds =20170220 and flag='True')C
	on A.tel =C.phone
	left join	
	wl_base.t_base_black_phone_fields D
	on substr(A.tel,0,7)=D.phone_field


wlcredit.t_credit_phone_tag_sogou

wl_base.t_base_yixin_daichang

wl_base.t_base_black_phone_fields
