'use strict';

const resource = require("./resource");

const buildResponse = (outputs, quickMain=true) => {
  return {
    statusCode: 200,
    body: JSON.stringify({
      version: "2.0",
      template: {
        outputs: outputs,
        quickReplies: (quickMain == true ? [getQuickReply("처음으로", "block", "처음으로", resource.welcomeBlockId)] : [])
      }
    })
  };
}

const buildAWSResponse = (body) => {
  return {
    statusCode: 200,
    body: JSON.stringify(body)
  };
}

const getQuickReply = (label, action, message, blockId=undefined, extra=undefined) => {
  const actionTypes = ["message", "block"];

  if (actionTypes.includes(action)) {
    var quickReply = {
      label: label,
      action: action,
      messageText: message
    };
    switch (action) {
      case "block": quickReply["blockId"] = blockId;
                    if (extra !== undefined) quickReply["extra"] = extra;
    }
    return quickReply;
  } else {
    throw "action type error";
  }
}

const getButton = (label, action, data, blockId=undefined, extra=undefined) => {
  const actionTypes = ["webLink", "osLink", "message", "phone", "share", "block"];

  if (actionTypes.includes(action)) {
    var button = {
      label: label,
      action: action
    };
    switch (action) {
      case "webLink": button["webLinkUrl"] = data; break;
      case "osLink": button["osLink"] = data; break;
      case "message": button["messageText"] = data; break;
      case "phone": button["phoneNumber"] = data; break;
      case "block": button["messageText"] = data;
                    button["blockId"] = blockId;
                    if (extra !== undefined) button["extra"] = extra;
    }
    return button;
  } else {
    throw "action type error";
  }
}

const getThumbnail = (imageUrl, fixedRatio=false, width=100, height=100) => {
  return {
    imageUrl: imageUrl,
    fixedRatio: fixedRatio,
    width: width,
    height: height
  };
}

const getListItem = (title, description=undefined, imageUrl=undefined, link=undefined) => {
  var listItem = {
    title: title,
  }
  if (description !== undefined) listItem["description"] = description;
  if (imageUrl !== undefined) listItem["imageUrl"] = imageUrl;
  if (link !== undefined) listItem["link"] = link;
  return listItem;
}

const getSimpleText = (text) => {
  return {
    simpleText: {
      text: text
    }
  };
}

const getSimpleImage = (imageUrl, altText="") => {
  return {
    simpleImage: {
        imageUrl: imageUrl,
        altText: altText
    }
  };
}

const getBasicCardBody = (title, description, thumbnail, buttons=[]) => {
  return {
    title: title,
    description: description,
    thumbnail: thumbnail,
    buttons: buttons
  };
}

const getBasicCard = (title, description, thumbnail, buttons=[]) => {
  return {
    basicCard: getBasicCardBody(title, description, thumbnail, buttons)
  };
}

const getListCard = (title, titleImageUrl, listItems, buttons=[]) => {
  return {
    listCard: {
      header: {
        title: title,
        imageUrl: titleImageUrl
      },
      items: listItems,
      buttons: buttons
    }
  };
}

const getCarousel = (type, cards) => {
  const cardTypes = ["basicCard", "commerceCard"];

  if (cardTypes.includes(type)) {
    return {
      carousel: {
        type: type,
        items: cards
      }
    };
  } else {
    throw "card type error";
  }
}

const getCommerceCardBody = (title, description, price, thumbnail, nickname, buttons=null) => {
  const body = {};
  if (title != null || description != null) {
    var str = (title != null ? title + "\n" : "");
    str += (description != null ? description : "");
    body["description"] = str;
  }
  if (price != null) {
    body["price"] = price;
    body["currency"] = "won"
  }
  if (thumbnail != null) {
    body["thumbnails"] = [ thumbnail ];
  }
  if (nickname != null) {
    body["profile"] = { nickname: nickname }
  }
  if (buttons != null) {
    body["buttons"] = buttons;
  }
  return body;
}

const getCommerceCard = (title, description, price, thumbnail, nickname, buttons=[]) => {
  return {
    commerceCard: getCommerceCardBody(title, description, price, thumbnail, nickname, buttons)
  };
}

module.exports = {
  buildResponse,
  buildAWSResponse,
  getButton,
  getThumbnail,
  getListItem,
  getSimpleText,
  getSimpleImage,
  getBasicCardBody,
  getBasicCard,
  getListCard,
  getCarousel,
  getCommerceCardBody,
  getCommerceCard
}
