-- user is send notifications only once and via only one channel of the 3 channels during this time period
-- Define a flag to identify when transaction happened; I am also including same day events 

WITH user_transaction_notf AS 
  (SELECT n.user_id,
             n.channel,
             n.created_date AS notification_date,
             u.birth_year,
             u.country,
             t.transaction_id,
             t.transactions_state,
             t.amount_usd,
             t.created_date AS transaction_date
      FROM notifications AS n
      LEFT JOIN users AS u ON n.user_id = u.user_id
      LEFT JOIN transactions AS t ON n.user_id = t.user_id
      WHERE n.status = 'SENT' -- Consider only successfully delivered notifications
      AND transactions_state = 'COMPLETED' -- filter only completed transaction)
      AND DATE(t.created_date) BETWEEN DATE(n.created_date) - 7 AND DATE(n.created_date) + 7
),

user_agg AS
  (SELECT user_id,
          channel,
          (2020- birth_year) AS user_age,
          country,
          amount_usd,
          transaction_id,
          CASE
              WHEN ((notification_date > transaction_date)
              --notification is after transaction
                    AND (extract(DAY FROM notification_date - transaction_date) BETWEEN 0 AND 7)) THEN 'before' 
              --transaction is after notification
              WHEN ((notification_date <= transaction_date)
                    AND (extract(DAY FROM notification_date - transaction_date) BETWEEN -7 AND 0)) THEN 'after' 

              ELSE 'others'
          END AS transaction_flag
   FROM user_transaction_notf
)

SELECT country,
       transaction_flag,
       CASE
           WHEN user_age BETWEEN 0 AND 20 THEN '0-20'
           WHEN user_age BETWEEN 21 AND 30 THEN '21-30'
           WHEN user_age BETWEEN 31 AND 40 THEN '31-40'
           WHEN user_age BETWEEN 41 AND 60 THEN '41-60'
           ELSE '>61'
       END AS age_group,
       count(DISTINCT user_id),
       count(DISTINCT transaction_id)/ count(DISTINCT user_id) AS transaction_per_user,
       sum(amount_usd)/ count(DISTINCT user_id) AS amount_per_user
FROM user_agg
WHERE transaction_flag IN ('after', 'before')
GROUP BY 1,
         2,
         3