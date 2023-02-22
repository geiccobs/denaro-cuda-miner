package main

import (
	"encoding/json"
	"fmt"
	"net"
	"time"

	"github.com/valyala/fasthttp"
)

var client = &fasthttp.Client{
	MaxConnDuration: time.Second * 300,
	ReadTimeout:     time.Second * 300,
	WriteTimeout:    time.Second * 300,
	Dial: func(addr string) (net.Conn, error) {
		return fasthttp.DialTimeout(addr, time.Second*5)
	},
}

func POST(url string, values map[string]interface{}) *fasthttp.Response {

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()

	req.SetRequestURI(url)
	req.SetConnectionClose()

	req.Header.SetMethod("POST")
	req.Header.SetContentType("application/json")

	kb, _ := json.Marshal(values)
	req.SetBody(kb)

	if err := client.Do(req, resp); err != nil {
		panic(err)
	}

	defer fasthttp.ReleaseRequest(req)

	return resp
}

func GET(url string, values map[string]interface{}) *fasthttp.Response {

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()

	url += "?"

	for key, value := range values {
		url += fmt.Sprintf("%s=%s&", key, value)
	}

	url = url[:len(url)-1]

	req.SetRequestURI(url)
	req.SetConnectionClose()

	req.Header.SetMethod("GET")
	req.Header.SetContentType("application/json")

	if err := client.Do(req, resp); err != nil {
		panic(err)
	}

	defer fasthttp.ReleaseRequest(req)

	return resp
}
