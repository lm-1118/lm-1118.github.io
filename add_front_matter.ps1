# PowerShell脚本：为所有缺少YAML前置元数据的文章添加元数据

$postsDir = ".\_posts"
$files = Get-ChildItem -Path $postsDir -Filter "*.md"

foreach ($file in $files) {
    $content = Get-Content -Path $file.FullName -Raw
    
    # 检查文件是否已经有YAML前置元数据
    if (-not ($content -match "^---\s*\r?\n")) {
        Write-Host "处理文件: $($file.Name)"
        
        # 提取标题（从文件名或文件内容中）
        $title = $file.BaseName -replace "^\d{4}-\d{2}-\d{2}-", ""
        $title = $title -replace "-", " "
        $title = (Get-Culture).TextInfo.ToTitleCase($title)
        
        # 根据文件名确定分类和标签
        $categories = @("未分类")
        $tags = @("笔记")
        
        # 根据文件名关键词设置分类和标签
        if ($file.Name -match "Go|golang") {
            $categories = @("编程", "Go")
            $tags = @("golang", "编程语言", "学习笔记")
        }
        elseif ($file.Name -match "sql") {
            $categories = @("数据库", "SQL")
            $tags = @("SQL", "数据库", "学习笔记")
        }
        elseif ($file.Name -match "计算机网络") {
            $categories = @("计算机基础", "网络")
            $tags = @("计算机网络", "学习笔记", "八股文")
        }
        elseif ($file.Name -match "日记") {
            $categories = @("个人", "日记")
            $tags = @("学习", "记录")
        }
        elseif ($file.Name -match "算法|hot100") {
            $categories = @("算法", "LeetCode")
            $tags = @("算法", "LeetCode", "编程题")
        }
        elseif ($file.Name -match "面试|面经") {
            $categories = @("求职", "面试")
            $tags = @("面试", "八股文", "求职")
        }
        elseif ($file.Name -match "项目") {
            $categories = @("项目", "开发")
            $tags = @("项目", "开发", "实践")
        }
        
        # 创建YAML前置元数据
        $categoriesStr = "[" + ($categories -join ", ") + "]"
        $tagsStr = "[" + ($tags -join ", ") + "]"
        $date = "2025-06-21 10:00:00 +0800"
        
        $yamlFrontMatter = @"
---
title: $title
date: $date
categories: $categoriesStr
tags: $tagsStr
---

"@
        
        # 添加YAML前置元数据到文件
        $newContent = $yamlFrontMatter + $content
        Set-Content -Path $file.FullName -Value $newContent -Encoding UTF8
        
        Write-Host "已添加YAML前置元数据到: $($file.Name)"
    }
    else {
        Write-Host "文件已有YAML前置元数据，跳过: $($file.Name)"
    }
}

Write-Host "处理完成!" 