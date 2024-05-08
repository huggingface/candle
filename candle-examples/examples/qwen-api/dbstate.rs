use redis::Client;
use sqlx::PgPool;
use sqlx::mysql::MySqlPool;
pub struct DbState {
    pub conn: MySqlPool,
    pub pconn: PgPool,
    pub redis_conn: Client,
}
