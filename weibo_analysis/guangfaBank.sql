#guangfa credit card
#sample table:wlservice.t_lt_guangfa_tel_snwb 
tel (string)
card (string)
snwb (string)
tb (string) 
tel_index
#rename column name
ALTER TABLE wlservice.t_lt_guangfa_tel_snwb CHANGE id1 card STRING

#1)join phone tag,paltform,black_phone_field of guangfa user
create table wlservice.t_lt_guangfa_user_info as
	select A.*,E.qqwb,E.qq,E.email,E.zhima_score,
	B.label as phone_tag,C.platform as daichang_platform,C.flag,
	D.phone_field as black_phone
	from
	wlservice.t_lt_guangfa_tel_snwb A
	left join 
	wlcredit.t_credit_phone_tag_sogou_label B
	on A.tel =B.phone
	left join
	(select * from wl_base.t_base_yixin_daichang
		where ds =20170220)C
	on A.tel =C.phone
	left join	
	wl_base.t_base_black_phone_fields D
	on substr(A.tel,0,7)=D.phone_field
	left join 
	wlcredit.t_zlj_zhima_model_train_lt E
	on A.tb=E.user_id

#2)extract weibo_usr_info data
create table wlservice.t_lt_guangfa_snwb_usrinfo as
	select A.tel,A.card,A.tb,B.*
	from
	wlservice.t_lt_guangfa_tel_snwb A
	left join
	wlservice.t_lt_subcol_weibo_usrinfo_all B
	on A.snwb = B.idstr

#3)extract weibo_text data
create table wlservice.t_lt_guangfa_snwb_text as
	select A.tel,A.card,A.tb,B.*
	from
	wlservice.t_lt_guangfa_tel_snwb A
	left join
	wlservice.t_lt_subcol_weibo_text_statistics_all B
	on A.snwb = B.user_id

#4)extract gaungfa follow data :622549247users
create table wlservice.t_lt_guangfa_snwb_follow as
	select B.id,B.ids
	from
	wlservice.t_lt_guangfa_tel_snwb A
	join
	(select * from wlbase_dev.t_base_weibo_user_fri 
		where ds=20161106)B
	on A.snwb = B.id


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

create table wl_service.t_lt_weibo_usr_followers as
	select friend as id, collect_set(snwb) as followers 
	from wl_service.t_lt_weibo_usr_fri_multline
	group by friend;

#old cluster test
create table wlservice.t_lt_weibo_usr_fri_100_multiline as
	select s.id,follow from wlservice.t_lt_weibo_usr_fri_100 s 
	lateral view explode(split(s.ids,',')) t as follow;

#http://blog.csdn.net/liluotuo/article/details/45673191
#http://jingyan.baidu.com/article/fcb5aff780a456edaa4a710c.html 
#lateral view用于和split、explode等UDTF一起使用的，能将一行数据拆分成多行数据，在此基础上可以对拆分的数据进行聚合

#5)dowload file

select s.*,sp from test.dual s 
lateral view explode(split(concat_ws(',','1','2','3','4','5','6','7','8','9'),',')) t as sp;