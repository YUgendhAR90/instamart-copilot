
-- Q1: WHERE and HAVING
-- Completed orders above INR 500
-- Customers whose qualifying order total exceeds INR 8000

SELECT
    customer_id,
    COUNT(*) AS qualifying_orders,
    ROUND(SUM(order_amount), 2) AS total_spend
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;


-- Q2: Three-table JOIN
-- Revenue by city and product category

SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN products p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;


-- Q3: Correlated subquery
-- Products priced above average actual selling price in their category

SELECT
    p.product_id,
    p.product_name,
    p.category,
    p.price_inr
FROM products p
WHERE p.price_inr > (
    SELECT AVG(o.order_amount * 1.0 / o.quantity)
    FROM orders o
    JOIN products p2
        ON p2.product_id = o.product_id
    WHERE p2.category = p.category
      AND o.status = 'completed'
)
ORDER BY p.category, p.price_inr DESC;



-- Q4: LEFT JOIN
-- Customers with no completed orders

SELECT
    c.customer_id,
    c.city,
    c.signup_date,
    c.age,
    c.gender
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
   AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- Q5: Common Table Expressions (CTEs)
-- Customers whose completed spending is above average

WITH customer_totals AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_totals AS (
    SELECT
        AVG(total_spend) AS average_spend
    FROM customer_totals
)
SELECT
    ct.customer_id,
    ROUND(ct.total_spend, 2) AS total_spend,
    ROUND(at.average_spend, 2) AS average_spend
FROM customer_totals ct
CROSS JOIN average_totals at
WHERE ct.total_spend > at.average_spend
ORDER BY ct.total_spend DESC;


-- Q6: Window functions
-- RANK, DENSE_RANK and ROW_NUMBER by city

WITH customer_totals AS (
    SELECT
        c.customer_id,
        c.city,
        SUM(o.order_amount) AS total_spend
    FROM customers c
    JOIN orders o
        ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
),
ranked_customers AS (
    SELECT
        customer_id,
        city,
        ROUND(total_spend, 2) AS total_spend,

        RANK() OVER (
            PARTITION BY city
            ORDER BY total_spend DESC
        ) AS rank_position,

        DENSE_RANK() OVER (
            PARTITION BY city
            ORDER BY total_spend DESC
        ) AS dense_rank_position,

        ROW_NUMBER() OVER (
            PARTITION BY city
            ORDER BY total_spend DESC
        ) AS row_number_position

    FROM customer_totals
)
SELECT *
FROM ranked_customers
ORDER BY city, row_number_position;


-- Q7: Running total
-- Daily completed revenue and cumulative revenue

WITH daily_revenue AS (
    SELECT
        substr(order_ts, 1, 10) AS day,
        SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT
    day,
    ROUND(revenue, 2) AS daily_revenue,
    ROUND(
        SUM(revenue) OVER (
            ORDER BY day
            ROWS BETWEEN UNBOUNDED PRECEDING
            AND CURRENT ROW
        ), 2
    ) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
