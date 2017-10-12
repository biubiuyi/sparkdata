#xianyu
create table wlservice.t_lt_xianyu_tmp as
	select user_id, avg(content_length) as avg_content_len,
	sum(annoy) as annoy_count,count(1) as buy_count
	from wl_analysis.t_base_record_cate_simple_xianyu
	group by user_id

#avg_buy_count per user
	161.83594529937076

#price distribution
select price,count(1) as price_count
from wl_analysis.t_base_record_cate_simple_xianyu
group by price
SORT BY price
#download

select min(price) as min_price,
percentile_approx(price,array(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)),
max(price) as max_price
from wl_analysis.t_base_record_cate_simple_xianyu

0
1.353612312680576
5.193740502527388
9.057256358355136
15.031578692962341
22.343936645568775
33.19713893073266
48.981721130391065
81.91288805801494
158.31063176470988
159700
#min 0	5754
#max 109000	125

#root_cat
create table wlservice.t_lt_xianyu_root_cat_distribution as
	select root_cat_id, root_cat_name, count(1) as root_buy_count
	from  wl_analysis.t_base_record_cate_simple_xianyu
	group by root_cat_id,root_cat_name



#monthall_active_score
select min(monthall_active_score) as min_monthall_active_score,
percentile_approx(monthall_active_score,array(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)),
max(monthall_active_score) as max_monthall_active_score
from wlcredit.t_credit_record_feature_online

min	0.27
0.1	0.27
0.2	0.3
0.3	0.35
0.4	0.41
0.5	0.49
0.6	0.5830958958
0.7	0.7223868112
0.8	0.94
0.9	1.37
max	28477.12

#wlbase_dev.t_base_user_profile
select verify, count(1) as verify_count 
from wlbase_dev.t_base_user_profile
group by verify
