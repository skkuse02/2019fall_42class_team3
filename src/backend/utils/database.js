'use strict';

const credential = require("./credential");
const resource = require("./resource");

const fs = require("fs");
const request = require("sync-request");

const aws = require("aws-sdk");
const s3 = new aws.S3({
  accessKeyId: credential.awsAccessKeyId,
  secretAccessKey: credential.awsSecretAccessKey,
  region: resource.awsS3Region
});

const mysql = require("mysql");
const pool = mysql.createPool({
  host: resource.mysqlLocation,
  user: credential.mysqlUser,
  password: credential.mysqlPassword,
  database: resource.mysqlName,
});
const query = async (sql, params) => {
    return await new Promise((resolve, reject) => {
        pool.query(sql, params, (error, results, fields) => {
            if (error) reject(error);
            else resolve(results);
            return;
        });
    });
};

const uploadImages = async (directory, locations) => {
  const results = [];
  for (var key in locations) {
    var location = locations[key];
    var filename = location.split("/");
    filename = filename[filename.length - 1].split("?")[0];
    if (location.includes("http")) {
      fs.writeFileSync("/tmp/" + filename, request("GET", location).getBody());
      location = "/tmp/" + filename;
    }

    const image = {
      Bucket: resource.awsS3Name,
      Key: directory + key + "-" + filename,
      ACL: "public-read",
      Body: fs.createReadStream(location)
    };

    const result = await new Promise((resolve, reject) => {
      s3.upload(image, (err, data) => {
        if (err) {
          resolve(null);
        } else {
          resolve(data.Location);
        }
      });
    });
    if (result != null) {
      results.push(result);
    }
  }
  return results;
};

const generateToken = (seed="") => {
  const tokenLength = 15;
  const tokenArray = ['0','1','2','3','4','5','6','7','8','9',
                      'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                      'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'];

  var token = "";
  for (var i = 1; i < tokenLength; i++) {
    token = token + tokenArray[Math.ceil(Math.random() * tokenArray.length) - 1];
  }

  return token;
};

const checkUserAuth = async (userId) => {
  const sql = 'SELECT * FROM user WHERE userId = ?';
  const user = await query(sql, [userId]);

  if (user.length == 0) {
    return null;
  } else {
    if (user[0].school_mail_auth == 1) {
      return user[0];
    } else {
      return null;
    }
  }
};

const registPendingAuthentication = async (userId, token) => {
  var sql = 'INSERT INTO authmail(userId, token)VALUES(?,?)';
  var params = [userId, token];

  try {
    const insert = await query(sql, params);
    console.log(insert);
    return {
      success: true,
      message: ""
    };
  } catch (error) {
    return {
      success: false,
      message: ""
    };
  }
};

const authPendingAuthentication = async (userId, token) => {
  var sql = 'UPDATE user SET school_mail_auth = 1 WHERE userId = ? and EXISTS (SELECT * FROM authmail WHERE authmail.userId = ?)';
  var params = [userId, userId];

  try{
    const update = await query(sql, params);
    console.log(update);
    return {
      success: true,
      message: ""
    };
  } catch(error){
    return{
      success: false,
      message: ""
    };
  }
};

const registNewUser = async (userId, nickname, school_name, school_mail) => {
  var sql = 'INSERT INTO user(userId,nickname, school_name, school_mail, school_mail_auth, openprofile, report_count, reliability_score)VALUES(?,?,?,?,?,?,?,?)';
  var params = [userId, nickname, school_name, school_mail, '0', null, 0, 50];

  try {
    const insert = await query(sql, params);
    console.log(insert);
    return {
      success: true,
      message: ""
    };
  } catch (error) {
    return {
      success: false,
      message: "이미 가입되어 있습니다."
    };
  }
};

const getUser = async (userId) => {
  var sql = 'SELECT * FROM user WHERE userId = ?';
  try {
    const pickuser = await query(sql, [userId]);
    console.log(pickuser);
    return {
      userId: pickuser[0].userId,
      nickname: pickuser[0].nickname,
      school_name: pickuser[0].school_name,
      school_mail: pickuser[0].school_mail,
      school_mail_auth: pickuser[0].school_mail_auth,
      openprofile: pickuser[0].openprofile,
      report_count: pickuser[0].report_count,
      reliability_score: pickuser[0].reliability_score
    };
  } catch (error) {
    return null;
  }
};

const registNewItem = async (userId, item_category, item_name, item_price, item_detail, item_image) => {
  const itemId = await generateToken();
  const imageurls = JSON.stringify(await uploadImages(itemId + '/', item_image));
  var sql = 'INSERT INTO item(userId, category, itemId, item_name, item_price, item_detail, item_image, item_date)VALUES(?,?,?,?,?,?,?,?)';
  var params = [userId, item_category, itemId, item_name, item_price, item_detail, imageurls, new Date().toISOString().slice(0, 19).replace('T', ' ')];
  console.log(Date.now());
  try {
    const insert = await query(sql, params);
    console.log(insert);
    return {
      success: true,
      message: ""
    };
  }catch(err){
    console.log(err);
    return{
      success: false,
      message:"fail"
    };
  }
};

const getItem = async (itemId) => {
  var sql = 'SELECT user.openprofile, user.reliability_score, user.nickname, item.* FROM item INNER JOIN user ON item.userId = user.userId WHERE itemId = ?';
  try{
    const pickitem = await query(sql, [itemId]);
    return {
      nickname: pickitem[0].nickname,
      userId: pickitem[0].userId,
      openprofile: pickitem[0].openprofile,
      item_category: pickitem[0].category,
      itemId: pickitem[0].itemId,
      item_name: pickitem[0].item_name,
      item_price: pickitem[0].item_price,
      item_detail: pickitem[0].item_detail,
      item_image: JSON.parse(pickitem[0].item_image),
      item_date: pickitem[0].item_date.toISOString().split("T")[0],
      reliability_score: pickitem[0].reliability_score
      };
  }catch(error){
    return null;
  }
};

const getUserItem = async (userId) => {
  const results = [];
  var sql = 'SELECT * FROM item WHERE userId = ?';
  try{
    const usersitem = await query(sql, [userId]);
    for (var i = 0; i<usersitem.length; i++){
      console.log(usersitem[i]);
      var res = {
        userId: usersitem[i].userId,
        item_category: usersitem[i].category,
        itemId: usersitem[i].itemId,
        item_name: usersitem[i].item_name,
        item_price: usersitem[i].item_price,
        item_detail: usersitem[i].item_detail,
        item_image: JSON.parse(usersitem[i].item_image),
        item_date: usersitem[i].item_date.toISOString().split("T")[0],
        reliability_score: usersitem[i].reliability_score
      };
      results.push(res);
    }
    return results;
  }catch(error){
    console.log(error);
    return null;
  }
};

const setOpenprofile = async (userId, openprofile) =>{
  var sql = 'UPDATE user SET openprofile = ? WHERE userId = ?';
  var params = [openprofile, userId];
  try {
    const update = await query(sql, params);
    console.log(update);
    return {
      success: true,
      message: ""
    };
  }catch(err){
    return{
      success: false,
      message:"fail"
    };
  }
};

const reportUser = async (userId) => {
  var sql = 'UPDATE user SET report_count = report_count + 1 WHERE userId = ?';
  var sql2 = 'SELECT report_count FROM user WHERE userId = ?';
  var params = [userId];
   try {
    const update = await query(sql, params);
     const getReportCount = await query(sql2, params);
     const report_count = getReportCount[0].report_count;
     if (report_count % 3 == 0){
       await downReliability(userId);
     }
    console.log(update);
    return {
      success: true,
      message: ""
    };
  }catch(err){
    return{
      success: false,
      message:"fail"
    };
  }
};

const downReliability = async(userId) => {
  var sql = 'UPDATE user SET reliability_score = reliability_score -30 WHERE userId = ?';
  var params = [userId];
  try {
    const update = await query(sql, params);
    console.log(update);
    return {
      success: true,
      message: ""
    };
  }catch(err){
    return{
      success: false,
      message:"fail"
    };
  }
};

const getCategory = async (category) => {
  const results = [];
  var sql = 'SELECT user.nickname, user.reliability_score, user.openprofile, item.* FROM user INNER JOIN item ON user.userId = item.userId WHERE item.category = ?';
  var params = [category];
  try{
    const searchitem = await query(sql, params);
    for (var i = 0; i<searchitem.length; i++){
      var res = {
        nickname: searchitem[i].nickname,
        openprofile: searchitem[i].openprofile,
        userId: searchitem[i].userId,
        item_category: searchitem[i].category,
        itemId: searchitem[i].itemId,
        item_name: searchitem[i].item_name,
        item_price: searchitem[i].item_price,
        item_detail: searchitem[i].item_detail,
        item_image: JSON.parse(searchitem[i].item_image),
        item_date: searchitem[i].item_date.toISOString().split("T")[0],
        reliability_score: searchitem[i].reliability_score
      };
      results.push(res);
    }
    return results;
  }catch(error){
    return null;
  }
};

const getKeyword = async (text) => {
  const results = [];
  var keyword = '%' + text + '%';
  var sql = 'SELECT user.nickname, user.reliability_score, user.openprofile, item.* FROM user INNER JOIN item ON user.userId = item.userId WHERE item.item_name LIKE ? OR item.item_detail LIKE ?';
  var params = [keyword, keyword];
  try{
    const searchitem = await query(sql, params);
    for (var i = 0; i<searchitem.length; i++){
      var res = {
        nickname: searchitem[i].nickname,
        openprofile: searchitem[i].openprofile,
        userId: searchitem[i].userId,
        category: searchitem[i].category,
        itemId: searchitem[i].itemId,
        item_name: searchitem[i].item_name,
        item_price: searchitem[i].item_price,
        item_detail: searchitem[i].item_detail,
        item_image: JSON.parse(searchitem[i].item_image),
        item_date: searchitem[i].item_date.toISOString().split("T")[0],
        reliability_score: searchitem[i].reliability_score
      };
      results.push(res);
    }
    return results;
  }catch(error){
    return null;
  }
};

const registTempDeal = async(userId, itemId, token) => {
  var sql = 'INSERT INTO temp_deal_list(userId, itemId, token)VALUES(?,?,?)';
  var params = [userId, itemId, token];
  try {
    const insert = await query(sql, params);
    console.log(insert);
    return {
      success: true,
      message: ""
    };
  }catch(err){
    console.log(err);
    return{
      success: false,
      message:"fail"
    };
  }
};

const registNewDeal = async(sell_id, buy_id, itemId, item_name) => {
  var sql = 'INSERT INTO deal_list(sell_id, buy_id, item_name)VALUES(?,?,?)';
  var params = [sell_id, buy_id, item_name];
  try {
    const insert = await query(sql, params);
    console.log(insert);
    await deleteItem(itemId);
    await upSellReliability(sell_id);
    await upBuyReliability(buy_id);
    return {
      success: true,
      message: ""
    };
  }catch(err){
    console.log(err);
    return{
      success: false,
      message:"fail"
    };
  }
};
const upSellReliability = async(sell_id) => {
  var sql = 'UPDATE user SET reliability_score = reliability_score + 5 WHERE userId = ?';
  var params = [sell_id];
  try {
    const update = await query(sql, params);
    console.log(update);
    return {
      success: true,
      message:""
    };
  }catch(err){
    console.log(err);
    return{
      success:false,
      message:"update fail"
    };
  }
};
const upBuyReliability = async(buy_id) => {
  var sql = 'UPDATE user SET reliability_score = reliability_score + 3 WHERE userId = ?';
  var params = [buy_id];
  try {
    const update = await query(sql, params);
    console.log(update);
    return {
      success: true,
      message:""
    };
  }catch(err){
    console.log(err);
    return{
      success:false,
      message:"update fail"
    };
  }
};
const deleteItem = async(itemId) => {
  var sql = 'DELETE FROM item WHERE itemId = ?';
  var params = [itemId];
  try {
    const deletion = await query(sql, params);
    console.log(deletion);
    return {
      success: true,
      message: ""
    };
  }catch(err){
    console.log(err);
    return{
      success: false,
      message:"fail"
    };
  }
};

const deleteUser = async(userId) => {
  var sql = 'DELETE FROM user WHERE userId = ?';
  var params = [userId];
  try {
    const deletion = await query(sql, params);
    console.log(deletion);
    return {
      success: true,
      message: ""
    };
  }catch(err){
    console.log(err);
    return{
      success: false,
      message:"fail"
    };
  }
};

const getTransaction = async(userId) => {
  const results = [];
  var sql = 'SELECT * FROM deal_list WHERE sell_id = ? or buy_id = ?';
  var params = [userId, userId];
  try{
    const getDeal = await query(sql, params);
    for (var i = 0; i<getDeal.length; i++){
      console.log(getDeal[i]);
      var res = {
        deal_num: getDeal[i].deal_num,
        sell_id : getDeal[i].sell_id,
        buy_id: getDeal[i].buy_id,
        item_name: getDeal[i].item_name
      };
      results.push(res);
    }
    return results;
  }catch(error){
    console.log(error);
    return null;
  }
};

module.exports = {
  uploadImages,
  checkUserAuth,
  registPendingAuthentication,
  authPendingAuthentication,
  registNewUser,
  getUser,
  registNewItem,
  getItem,
  getUserItem,
  getCategory,
  getKeyword,
  registTempDeal,
  registNewDeal,
  deleteItem,
  deleteUser,
  getTransaction,
  setOpenprofile,
  reportUser
}
