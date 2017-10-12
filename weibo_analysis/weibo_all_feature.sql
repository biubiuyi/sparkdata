#1)weibo usrinfo,drop_duplicates/ ds=20161213
create table wlservice.t_lt_subcol_weibo_usrinfo_all as
	SELECT
	idstr,screen_name,name,province,city,location,description,gender,
	followers_count,friends_count,statuses_count,
	favourites_count,created_at,verified,online_status,
	bi_followers_count,
	(bi_followers_count/friends_count) as bi_friends_ratio,
    (bi_followers_count/followers_count) as bi_followers_ratio,
    (friends_count/followers_count) as friends_followers_ratio
	from
	wlbase_dev.t_base_weibo_user_new 
	where ds=20161213

#2)weibo text:t_lt_subcol_weibo_text_statistics_all
#=================================================================
#1))ds =20161126
create table wlservice.t_lt_subcol_weibo_text_all as
	SELECT
	user_id,mid,created_at,text,source,thumbnail_pic,reposts_count,
	comments_count,attitudes_count,weibo_type,islongtext,retweeted_status,
	(case when retweeted_status='-' then 1 else 0 end) as is_original
	from
	wlbase_dev.t_base_weibo_text 
	where ds=20161126

#2))weibo text statistics 
create table wlservice.t_lt_subcol_weibo_text_statistics_all_new as
	select user_id,
	avg(length(text)) as weibo_avglen,
	sum(reposts_count) as sum_reposts,
	sum(comments_count ) as sum_comments,
	sum(attitudes_count) as sum_attitudes,
	sum(case when thumbnail_pic='-' then 0 else 1 end) as sum_picture,
	sum(islongtext) as sum_longtext,
	sum(is_original) as sum_original,
	avg(reposts_count) as avg_reposts,
	avg(comments_count ) as avg_comments,
	avg(attitudes_count) as avg_attitudes,
	avg(islongtext) as ratio_longtext,
	sum(weibo_type) as weibo_special,
	count(1) as count_weibo
	from wlservice.t_lt_subcol_weibo_text_all
	group by user_id;

create table wlservice.t_lt_subcol_weibo_content_time_str as
	select user_id,	
	collect_list(text) as contents_str
	from wlservice.t_lt_subcol_weibo_text_all
	group by user_id
create table wlservice.t_lt_subcol_weibo_time_str as
	select user_id,
	collect_list(created_at) as times_str
	from wlservice.t_lt_subcol_weibo_text_all
	group by user_id
create table wlservice.t_lt_subcol_weibo_source_count as
	select user_id,
	size(collect_set(source)) as source_count
	from wlservice.t_lt_subcol_weibo_text_all
	group by user_id

#3))multiple table join
create table wlservice.t_lt_subcol_weibo_text_statistics_all as
	select D.*, A.source_count,B.times_str,C.contents_str
	from 
	wlservice.t_lt_subcol_weibo_source_count A
	join
	wlservice.t_lt_subcol_weibo_time_str B
	on (A.user_id=B.user_id)
	join
	wlservice.t_lt_subcol_weibo_content_time_str C
	on C.user_id=B.user_id
	join 
	wlservice.t_lt_subcol_weibo_text_statistics_all_new D
	on C.user_id=D.user_id
#4)) drop tmp table A,B,C
#=================================================================

#all snwb usr follow data :
#base DB:wlbase_dev.t_base_weibo_user_fri /ds = 20161106
#final:t_lt_weibo_usr_follows_all
create table wlservice.t_lt_weibo_usr_follows_all as
SELECT D.id,D.ids from 
(select * from 
	wlbase_dev.t_base_weibo_user_fri  where ds=20161106) D
join
(select case when A.idstr is null then B.user_id else A.idstr end 
	as id from 
(select idstr from wlservice.t_lt_subcol_weibo_usrinfo_all) A
		FULL OUTER JOIN
		(select user_id 
			from wlservice.t_lt_subcol_weibo_text_statistics_all) B
			on A.idstr = B.user_id) C
on D.id = C.id

==============weibo followers generate===================
#split the array to multiple line :new cluster
create table wl_service.t_lt_weibo_usr_fri_multline as
	select id as snwb,friend from wl_base.t_base_weibo_user_fri 	 
	lateral view explode(split(ids,',')) t as friend
	where ds=20161106;


create table wl_service.t_lt_weibo_usr_followers as
	select friend as id, collect_set(snwb) as followers 
	from wl_service.t_lt_weibo_usr_fri_multline
	group by friend;


#extract all qq weibo userinfo and qq_usrinfo
create table wlservice.t_lt_qq_weibo_user_info as
	select D.*,E.gender_qq,E.nick,E.phone,E.college,E.shengxiao,E.occupation,
	E.constel,E.blood,E.email_qq,E.age,E.birthday,E.loc from 
	(select A.*,B.gender as gender_wb,B.isvip,B.auth,B.nickname as nick_wb,
		B.regtime,
		concat_ws('_',B.b_year,B.b_month,B.b_day) as birthday_wb,
		concat_ws('_',B.l_countryname,B.l_provincename,B.l_cityname) as loc_wb
		from
		wlcredit.t_zlj_zhima_model_train_lt A
		left join
		wlbase_dev.t_base_qq_weibo_user_info B
		on A.qqwb = B.id)D
	left join
	(select A.user_id, C.gender as gender_qq,C.nick,C.phone,C.college,
		C.shengxiao,C.occupation,
		C.constel,C.blood,C.email as email_qq,C.age,C.birthday,C.loc
		from
		wlcredit.t_zlj_zhima_model_train_lt A
		left join
		(select * from wlbase_dev.t_base_qq_user_dev where ds=20160923)C
		on A.qq = C.uin)E	
	on D.user_id=E.user_id
	
